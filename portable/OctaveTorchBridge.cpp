// In-process Octave bridge to the Python/Torch GIF energy evaluator.
//
// This .oct extension replaces the retired pipe worker/serialization path: it
// embeds CPython in the Octave process and reuses the analytic evaluator in
// BCP/GIF_and_CM/torch_energy.py (the Torch port of symmetricDirichletEnergy.m).
// There is no LibTorch/CUDA formula duplicate and no second process.
//
// Data movement contract:
//  - 'prepare' wraps the Octave input matrices as Fortran-order NumPy views
//    (no host copy) and performs the original once-per-remesh host-to-device
//    upload; per-step movement is limited to the free coordinate vector x.
//  - 'evaluate' runs the device computation first, reads the device-result
//    metadata (foldover/kept-face counts can vary per call), then allocates
//    the Octave output arrays, wraps their buffers as NumPy views, and lets
//    the evaluator copy GPU results directly into those buffers.  No
//    evaluate()->NumPy allocation followed by an O(F) memcpy into Octave
//    arrays.  A column-major Hessian destination is filled by a device-side
//    transpose gather plus a single contiguous device-to-host copy; no
//    host-side transpose intermediate is created.
//  - Downloads complete synchronously before results are returned to Octave.
//
// The interpreter and CUDA are initialized lazily: a 'close' with no live
// evaluator (and the dispatcher's CPU reference fallback) never touch Python.

#if defined(_WIN32)
#error "OctaveTorchBridge is POSIX-only, matching the optional Octave backend."
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <oct.h>
#include <defun-dld.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

namespace {

constexpr char kTorchEvaluatorErrorId[] = "BCP:TorchEvaluator";

struct bridge_state {
    PyObject* evaluator = nullptr;  // torch_energy.TorchEnergy instance (owned)
    std::string device;
    std::int64_t prepare_count = 0;
    std::int64_t evaluate_count = 0;
    std::int64_t matrix_upload_bytes = 0;
    std::int64_t host_download_bytes = 0;
};

bridge_state g_state;
bool g_numpy_ready = false;

[[noreturn]] void fail(const char* id, const std::string& message) {
    error_with_id(id, "%s", message.c_str());
    throw std::runtime_error(message);  // unreachable; error_with_id throws
}

struct gil_guard {
    PyGILState_STATE state;
    gil_guard() : state(PyGILState_Ensure()) {}
    ~gil_guard() { PyGILState_Release(state); }
    gil_guard(const gil_guard&) = delete;
    gil_guard& operator=(const gil_guard&) = delete;
};

// Owned PyObject* whose DECREF runs under the enclosing gil_guard (declare the
// guard before any py_ref so destruction order keeps the GIL held).
struct py_ref {
    PyObject* object;
    py_ref() : object(nullptr) {}
    explicit py_ref(PyObject* owned) : object(owned) {}
    ~py_ref() { Py_XDECREF(object); }
    py_ref(const py_ref&) = delete;
    py_ref& operator=(const py_ref&) = delete;
    py_ref(py_ref&& other) noexcept : object(other.object) { other.object = nullptr; }
    py_ref& operator=(py_ref&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(object);
            object = other.object;
            other.object = nullptr;
        }
        return *this;
    }
    PyObject* get() const { return object; }
    PyObject* release() { PyObject* result = object; object = nullptr; return result; }
    explicit operator bool() const { return object != nullptr; }
};

// Fetch (which clears) the current Python error as "TypeName: message".
std::string python_error_string() {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    std::string message = "unknown Python error";
    if (type != nullptr) {
        py_ref type_guard(type);
        py_ref value_guard(value);
        py_ref traceback_guard(traceback);
        py_ref name(PyObject_GetAttrString(type, "__name__"));
        if (name) {
            const char* text = PyUnicode_AsUTF8(name.get());
            if (text != nullptr) message = text;
        }
        if (value != nullptr) {
            py_ref text(PyObject_Str(value));
            if (text) {
                const char* utf8 = PyUnicode_AsUTF8(text.get());
                if (utf8 != nullptr) message += ": " + std::string(utf8);
            }
        }
    }
    PyErr_Clear();
    return message;
}

void ensure_python_initialized() {
    if (Py_IsInitialized()) return;
    Py_InitializeEx(0);  // 0: leave process signal handlers to Octave.
    if (!Py_IsInitialized())
        fail("BCP:TorchRuntime", "Failed to initialize the embedded Python interpreter.");
}

int import_numpy_api() {
    import_array1(-1);  // returns -1 on failure instead of returning void
    return 0;
}

void ensure_numpy() {
    if (g_numpy_ready) return;
    if (import_numpy_api() < 0)
        fail("BCP:TorchRuntime", "NumPy is required by the Torch evaluator but could not be imported.");
    g_numpy_ready = true;
}

void reset_counters() {
    g_state.prepare_count = 0;
    g_state.evaluate_count = 0;
    g_state.matrix_upload_bytes = 0;
    g_state.host_download_bytes = 0;
}

// Release the evaluator and its device cache.  Call with the GIL held.
void release_evaluator() {
    Py_XDECREF(g_state.evaluator);
    g_state.evaluator = nullptr;
    g_state.device.clear();
    reset_counters();
}

void delete_octave_value_capsule(PyObject* capsule) {
    delete static_cast<octave_value*>(PyCapsule_GetPointer(capsule, "bcp_octave_value"));
}

void delete_staging_capsule(PyObject* capsule) {
    std::free(PyCapsule_GetPointer(capsule, "bcp_staging"));
}

PyObject* wrap_with_capsule(void* data, octave_idx_type rows, octave_idx_type cols,
                            int type_num, npy_intp itemsize, PyObject* capsule) {
    npy_intp dims[2] = {rows, cols};
    npy_intp strides[2] = {itemsize, itemsize * rows};
    py_ref array(PyArray_New(&PyArray_Type, 2, dims, type_num, strides, data, 0,
                             NPY_ARRAY_WRITEABLE, nullptr));
    if (!array) {
        Py_DECREF(capsule);
        fail("BCP:TorchRuntime", python_error_string());
    }
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject*>(array.get()),
                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(array.get()), capsule) < 0) {
        Py_DECREF(capsule);
        fail("BCP:TorchRuntime", python_error_string());
    }
    return array.release();
}

// Wrap an existing dense buffer as a Fortran-order NumPy view (no copy).  The
// capsule base keeps the owning octave_value alive while Python holds the view.
PyObject* wrap_buffer(void* data, octave_idx_type rows, octave_idx_type cols,
                      int type_num, npy_intp itemsize, const octave_value& owner) {
    // PyArray_New with null data allocates its own buffer, which conflicts
    // with the ownership capsule; empty views use a never-touched placeholder.
    static std::int64_t empty_placeholder[2] = {0, 0};
    if (rows * cols == 0) data = empty_placeholder;
    auto* holder = new octave_value(owner);  // shares the buffer (copy-on-write)
    PyObject* capsule = PyCapsule_New(holder, "bcp_octave_value", &delete_octave_value_capsule);
    if (capsule == nullptr) {
        delete holder;
        fail("BCP:TorchRuntime", python_error_string());
    }
    return wrap_with_capsule(data, rows, cols, type_num, itemsize, capsule);
}

// Wrap a caller-owned staging buffer (freed with the view).  Used for the
// single-element values that Octave stores as scalars with no shareable
// buffer; this is never used for O(F) data.
PyObject* wrap_staging(void* data, octave_idx_type rows, octave_idx_type cols,
                       int type_num, npy_intp itemsize) {
    PyObject* capsule = PyCapsule_New(data, "bcp_staging", &delete_staging_capsule);
    if (capsule == nullptr) {
        std::free(data);
        fail("BCP:TorchRuntime", python_error_string());
    }
    return wrap_with_capsule(data, rows, cols, type_num, itemsize, capsule);
}

// Wrap a dense double/complex-double Octave matrix as a NumPy view (no copy).
// The accepted types are exactly the ones the retired pipe transport carried.
PyObject* wrap_octave_matrix(const octave_value& value) {
    const std::string type = value.type_name();
    if (value.iscomplex()) {
        if (type == "complex scalar") {  // no shareable buffer; stage 1 element
            auto* staged = static_cast<std::complex<double>*>(std::malloc(sizeof(std::complex<double>)));
            if (staged == nullptr) fail("BCP:TorchRuntime", "Out of memory staging a complex scalar.");
            *staged = value.complex_value();
            return wrap_staging(staged, 1, 1, NPY_COMPLEX128, 16);
        }
        if (type != "complex matrix")
            fail("BCP:TorchProtocol", "Expected a dense complex matrix, got " + type + ".");
        const ComplexMatrix matrix = value.complex_matrix_value();
        return wrap_buffer(const_cast<std::complex<double>*>(matrix.data()),
                           matrix.rows(), matrix.cols(), NPY_COMPLEX128, 16, value);
    }
    if (type == "scalar") {  // no shareable buffer; stage 1 element
        auto* staged = static_cast<double*>(std::malloc(sizeof(double)));
        if (staged == nullptr) fail("BCP:TorchRuntime", "Out of memory staging a scalar.");
        *staged = value.double_value();
        return wrap_staging(staged, 1, 1, NPY_DOUBLE, 8);
    }
    if (type != "matrix" && type != "null_matrix")  // [] is a null_matrix
        fail("BCP:TorchProtocol", "Expected a dense real matrix, got " + type + ".");
    const Matrix matrix = value.matrix_value();
    return wrap_buffer(const_cast<double*>(matrix.data()),
                       matrix.rows(), matrix.cols(), NPY_DOUBLE, 8, value);
}

bool bool_argument(const octave_value& value, const char* name) {
    if (value.islogical() && value.numel() == 1) return value.bool_value();
    if (value.isnumeric() && value.numel() == 1) return value.double_value() != 0;
    fail("BCP:TorchProtocol", std::string("'") + name + "' must be a logical scalar.");
}

octave_scalar_map stats_map() {
    octave_scalar_map meta;
    meta.assign("device", octave_value(g_state.device));
    meta.assign("backend", octave_value(std::string("inprocess")));
    meta.assign("pid", octave_value(static_cast<double>(getpid())));
    meta.assign("prepare_count", octave_value(static_cast<double>(g_state.prepare_count)));
    meta.assign("evaluate_count", octave_value(static_cast<double>(g_state.evaluate_count)));
    meta.assign("matrix_upload_bytes", octave_value(static_cast<double>(g_state.matrix_upload_bytes)));
    meta.assign("host_download_bytes", octave_value(static_cast<double>(g_state.host_download_bytes)));
    return meta;
}

void prepend_module_dir(const std::string& module_dir) {
    PyObject* path = PySys_GetObject("path");  // borrowed
    if (path == nullptr || !PyList_Check(path)) return;
    py_ref dir(PyUnicode_FromString(module_dir.c_str()));
    if (!dir) fail("BCP:TorchRuntime", python_error_string());
    const int contained = PySequence_Contains(path, dir.get());
    if (contained < 0) fail("BCP:TorchRuntime", python_error_string());
    if (contained == 0 && PyList_Insert(path, 0, dir.get()) < 0)
        fail("BCP:TorchRuntime", python_error_string());
}

std::pair<octave_idx_type, octave_idx_type> device_result_shape(PyObject* shapes, const char* name) {
    PyObject* entry = PyDict_GetItemString(shapes, name);  // borrowed
    if (entry == nullptr || !PyTuple_Check(entry) || PyTuple_GET_SIZE(entry) != 2)
        fail("BCP:TorchProtocol",
             std::string("Malformed device-result metadata for output '") + name + "'.");
    const long long rows = PyLong_AsLongLong(PyTuple_GET_ITEM(entry, 0));
    const long long cols = PyLong_AsLongLong(PyTuple_GET_ITEM(entry, 1));
    if (PyErr_Occurred() || rows < 0 || cols < 0)
        fail("BCP:TorchProtocol",
             std::string("Malformed device-result shape for output '") + name + "'.");
    return {static_cast<octave_idx_type>(rows), static_cast<octave_idx_type>(cols)};
}

PyObject* call_method(PyObject* receiver, const char* method, PyObject* arguments) {
    py_ref callable(PyObject_GetAttrString(receiver, method));
    if (!callable) fail(kTorchEvaluatorErrorId, python_error_string());
    return PyObject_CallObject(callable.get(), arguments);
}

octave_value_list do_start(const octave_value_list& args) {
    if (args.length() < 2 || !args(1).is_string())
        fail("BCP:TorchProtocol", "bcp_torch_energy_native: 'start' requires a device string.");
    const std::string requested = args(1).string_value();
    std::string module_dir;
    if (args.length() >= 3 && args(2).is_string()) module_dir = args(2).string_value();
    if (module_dir.empty()) {
        const char* environment = std::getenv("BCP_TORCH_MODULE_DIR");
        if (environment != nullptr) module_dir = environment;
    }
    if (g_state.evaluator != nullptr && g_state.device == requested)
        return octave_value_list(1, octave_value(stats_map()));
    if (g_state.evaluator != nullptr) {
        gil_guard gil;
        release_evaluator();
    }
    ensure_python_initialized();
    gil_guard gil;
    ensure_numpy();
    if (!module_dir.empty()) prepend_module_dir(module_dir);
    py_ref module(PyImport_ImportModule("torch_energy"));
    if (!module)
        fail("BCP:TorchRuntime", "Cannot import torch_energy: " + python_error_string());
    py_ref device_argument(PyUnicode_FromString(requested.c_str()));
    if (!device_argument) fail("BCP:TorchRuntime", python_error_string());
    py_ref tuple(PyTuple_Pack(1, device_argument.get()));
    if (!tuple) fail("BCP:TorchRuntime", python_error_string());
    PyObject* instance = call_method(module.get(), "TorchEnergy", tuple.get());
    if (instance == nullptr)
        fail(kTorchEvaluatorErrorId, python_error_string());
    g_state.evaluator = instance;
    g_state.device = requested;
    reset_counters();
    return octave_value_list(1, octave_value(stats_map()));
}

octave_value_list do_prepare(const octave_value_list& args) {
    if (g_state.evaluator == nullptr)
        fail("BCP:TorchProcess", "The Torch evaluator must be started before prepare/evaluate.");
    if (args.length() != 6 && args.length() != 7)
        fail("BCP:TorchProtocol",
             "bcp_torch_energy_native: 'prepare' expects J_fz, J_fbz, area, fixed_indices, "
             "fixed_values[, hessian_indices].");
    gil_guard gil;
    const int value_count = args.length() - 1;
    py_ref tuple(PyTuple_New(value_count));
    if (!tuple) fail("BCP:TorchRuntime", python_error_string());
    std::int64_t upload_bytes = 0;
    for (int k = 0; k < value_count; ++k) {
        PyObject* view = wrap_octave_matrix(args(k + 1));
        upload_bytes += PyArray_NBYTES(reinterpret_cast<PyArrayObject*>(view));
        PyTuple_SET_ITEM(tuple.get(), k, view);  // steals the reference
    }
    py_ref result(call_method(g_state.evaluator, "prepare", tuple.get()));
    if (!result) {
        // A failed prepare must not leave a usable stale cache behind.
        release_evaluator();
        fail(kTorchEvaluatorErrorId, python_error_string());
    }
    g_state.prepare_count += 1;
    g_state.matrix_upload_bytes += upload_bytes;
    return octave_value_list();
}

octave_value_list do_evaluate(const octave_value_list& args, int nargout) {
    if (g_state.evaluator == nullptr)
        fail("BCP:TorchProcess", "The Torch evaluator must be started before prepare/evaluate.");
    if (args.length() != 4)
        fail("BCP:TorchProtocol",
             "bcp_torch_energy_native: 'evaluate' expects x, spd, exclude_flips.");
    if (nargout > 6)
        fail("BCP:TorchProtocol", "bcp_torch_energy_native: 'evaluate' returns at most 6 outputs.");
    static const char* kNames[6] = {"foldovers", "fz", "fbz", "E", "G", "H"};
    const int output_count = std::max(0, std::min(nargout, 6));
    gil_guard gil;
    py_ref x_view(wrap_octave_matrix(args(1)));
    py_ref spd(PyBool_FromLong(bool_argument(args(2), "spd") ? 1 : 0));
    py_ref exclude(PyBool_FromLong(bool_argument(args(3), "exclude_flips") ? 1 : 0));
    py_ref call(PyTuple_Pack(3, x_view.get(), spd.get(), exclude.get()));
    py_ref evaluation(call_method(g_state.evaluator, "evaluate_device", call.get()));
    if (!evaluation) {
        release_evaluator();
        fail(kTorchEvaluatorErrorId, python_error_string());
    }
    py_ref shapes(call_method(evaluation.get(), "output_shapes", nullptr));
    if (!shapes || !PyDict_Check(shapes.get())) {
        release_evaluator();
        fail("BCP:TorchProtocol", "The Torch evaluator returned malformed output metadata.");
    }
    octave_value_list results(output_count);
    py_ref outputs(PyDict_New());
    if (!outputs) fail("BCP:TorchRuntime", python_error_string());
    std::int64_t download_bytes = 0;
    // Octave stores a 1x1 result as a (complex) scalar whose buffer cannot be
    // shared, so single-element outputs are staged through one heap element
    // (freed by the view's capsule after the download) and materialized
    // afterwards; this never applies to O(F) data.
    void* staged[6] = {};
    for (int k = 0; k < output_count; ++k) {
        const bool complex_output = (k == 1 || k == 2);  // fz, fbz
        const npy_intp itemsize = complex_output ? 16 : 8;
        const int type_num = complex_output ? NPY_COMPLEX128 : NPY_DOUBLE;
        const auto [rows, cols] = device_result_shape(shapes.get(), kNames[k]);
        py_ref view;
        if (rows == 1 && cols == 1) {
            staged[k] = std::calloc(1, itemsize);
            if (staged[k] == nullptr)
                fail("BCP:TorchRuntime", "Out of memory staging a scalar output.");
            view = py_ref(wrap_staging(staged[k], 1, 1, type_num, itemsize));
        } else {
            octave_value value;
            void* data = nullptr;
            if (complex_output) {
                ComplexMatrix matrix(rows, cols);
                // An all-real ComplexMatrix is squeezed to real storage when
                // the octave_value is created, which would change the output
                // type and detach it from the wrapped buffer; keep it complex.
                if (rows * cols > 0) matrix(0, 0) = std::complex<double>(0.0, 1.0);
                value = octave_value(matrix);  // shares the buffer, stays complex
                const ComplexMatrix shared = value.complex_matrix_value();
                data = const_cast<std::complex<double>*>(shared.data());
                if (rows * cols > 0)
                    std::memset(data, 0,
                                static_cast<size_t>(rows) * cols * sizeof(std::complex<double>));
            } else {
                value = octave_value(Matrix(rows, cols));  // shares the buffer
                const Matrix shared = value.matrix_value();
                data = const_cast<double*>(shared.data());
            }
            view = py_ref(wrap_buffer(data, rows, cols, type_num, itemsize, value));
            results(k) = value;
        }
        if (PyDict_SetItemString(outputs.get(), kNames[k], view.get()) < 0)
            fail("BCP:TorchRuntime", python_error_string());
        download_bytes += static_cast<std::int64_t>(rows) * cols * itemsize;
    }
    py_ref write_arguments(PyTuple_Pack(1, outputs.get()));
    py_ref written(call_method(evaluation.get(), "write_into", write_arguments.get()));
    if (!written) {
        release_evaluator();
        fail(kTorchEvaluatorErrorId, python_error_string());
    }
    for (int k = 0; k < output_count; ++k) {
        if (staged[k] != nullptr)
            results(k) = (k == 1 || k == 2)
                             ? octave_value(*static_cast<std::complex<double>*>(staged[k]))
                             : octave_value(*static_cast<double*>(staged[k]));
    }
    g_state.evaluate_count += 1;
    g_state.host_download_bytes += download_bytes;
    return results;
}

// Release the evaluator when the .oct is unloaded (clear/autoload or exit).
// The interpreter itself is left running: finalizing embedded CPython with
// live Torch state inside a host process teardown is not reliable.
struct bridge_finalizer {
    ~bridge_finalizer() {
        if (Py_IsInitialized() && g_state.evaluator != nullptr) {
            gil_guard gil;
            release_evaluator();
        }
    }
};

bridge_finalizer g_finalizer;

}  // namespace

DEFUN_DLD(bcp_torch_energy_native, args, nargout,
          "In-process Torch GIF energy evaluator (start/prepare/evaluate/close/stats).\n"
          "Reuses BCP/GIF_and_CM/torch_energy.py; normally driven by bcp_torch_energy.m.") {
    if (args.length() < 1 || !args(0).is_string())
        error_with_id("BCP:TorchProtocol",
                      "bcp_torch_energy_native: the first argument must be an operation string.");
    const std::string operation = args(0).string_value();
    if (operation == "close") {
        // A no-op close never initializes Python or CUDA.
        if (g_state.evaluator != nullptr) {
            gil_guard gil;
            release_evaluator();
        }
        return octave_value_list();
    }
    if (operation == "start") return do_start(args);
    if (operation == "prepare") return do_prepare(args);
    if (operation == "evaluate") return do_evaluate(args, nargout);
    if (operation == "stats") {
        if (g_state.evaluator == nullptr)
            fail("BCP:TorchProcess", "The Torch evaluator must be started before stats.");
        return octave_value_list(1, octave_value(stats_map()));
    }
    error_with_id("BCP:TorchProtocol", "Unknown Torch evaluator operation: %s", operation.c_str());
    return octave_value_list();
}
