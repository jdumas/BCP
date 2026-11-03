// Octave MAT transport regression: borrowed setter buffers, direct getter
// destinations, row/column-major and numeric/complex conversions, size/rank
// error paths and private temporary-file cleanup. argv[1] is a work directory.
#include "Utils/MatlabInterface.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Complex = std::complex<double>;

namespace {
void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::string thrownMessage(const std::function<void()>& call) {
    try {
        call();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

// Every MAT transfer file lives in this engine's private bcp-octave-* directory.
size_t transferFileCount(const fs::path& tmp) {
    size_t count = 0;
    for (const auto& dir : fs::directory_iterator(tmp))
        if (dir.is_directory() && dir.path().filename().string().rfind("bcp-octave-", 0) == 0)
            for (const auto& file : fs::directory_iterator(dir.path())) (void)file, ++count;
    return count;
}

double element(size_t i, size_t j) { return 0.5 + double(i) * 10.0 - double(j) / 3.0; }

void testRealSetters(MatlabInterface& engine) {
    // Column-major double input is borrowed by matio for the write; changing the
    // caller buffer after the call must not affect Octave's copy.
    const unsigned rows = 3, cols = 4;
    std::vector<double> colMajor(rows * cols), rowMajor(rows * cols);
    for (unsigned j = 0; j < cols; ++j)
        for (unsigned i = 0; i < rows; ++i) {
            colMajor[j * rows + i] = element(i, j);
            rowMajor[i * cols + j] = element(i, j);
        }
    const std::vector<double> original = colMajor;
    check(!engine.SetEngineRealMatrix("borrowed", rows, cols, colMajor.data(), true), "Borrowed set failed");
    check(colMajor == original, "Setter modified the caller's buffer");
    std::fill(colMajor.begin(), colMajor.end(), -1.0);
    check(!engine.SetEngineRealMatrix("converted", rows, cols, rowMajor.data(), false), "Row-major set failed");
    engine.Eval("[ii, jj] = ndgrid(0:2, 0:3); expected = 0.5 + ii * 10 - jj / 3;"
                "assert(isequal(borrowed, expected)); assert(isequal(converted, expected));"
                "assert(isa(borrowed, 'double') && isreal(borrowed));");

    // Null values create zeros; empty matrices keep their shape.
    check(!engine.SetEngineRealMatrix<double>("zeros_set", 2, 3, nullptr, true), "Null-value set failed");
    check(!engine.SetEngineRealMatrix<double>("empty_set", 0, 3, colMajor.data(), true), "Empty set failed");
    engine.Eval("assert(isequal(zeros_set, zeros(2,3))); assert(isequal(size(empty_set), [0 3]));");

    // Large borrowed transfer: exact round trip through the direct getter.
    const unsigned largeRows = 100003, largeCols = 3;
    std::vector<double> large(size_t(largeRows) * largeCols), back(large.size(), 0.0);
    for (size_t k = 0; k < large.size(); ++k) large[k] = std::sin(double(k)) * 1e3 + 1.0 / (k + 1.0);
    check(!engine.SetEngineRealMatrix("large", largeRows, largeCols, large.data(), true), "Large set failed");
    check(!engine.GetEngineRealMatrix("large", largeRows, largeCols, back.data(), true), "Large get failed");
    check(back == large, "Large borrowed transfer changed values");

    // Sparse triplets are staged once and borrowed for the dense transfer.
    const unsigned ri[] = {0, 2, 1}, ci[] = {1, 0, 2};
    const double sv[] = {2.5, -1.0, 4.0};
    check(!engine.SetEngineSparseRealMatrix("sp", 3, ri, ci, sv, 3, 3), "Sparse set failed");
    engine.Eval("assert(issparse(sp) && isequal(full(sp), [0 2.5 0; 0 0 4; -1 0 0]));");
}

void testComplexSetters(MatlabInterface& engine) {
    const Complex colMajor[] = {{1, 2}, {3, -4}, {-5, 6}, {7, 0.5}, {0, -1}, {2.5, 2.5}};
    Complex rowMajor[6];
    for (unsigned i = 0; i < 2; ++i)
        for (unsigned j = 0; j < 3; ++j) rowMajor[i * 3 + j] = colMajor[j * 2 + i];
    check(!engine.SetEngineComplexMatrix("zc", 2, 3, colMajor, true), "Complex column-major set failed");
    check(!engine.SetEngineComplexMatrix("zr", 2, 3, rowMajor, false), "Complex row-major set failed");
    engine.Eval("expected = [1+2i, -5+6i, 0-1i; 3-4i, 7+0.5i, 2.5+2.5i];"
                "assert(isequal(zc, expected)); assert(isequal(zr, expected));");
    check(!engine.SetEngineComplexMatrix<double>("zempty", 0, 0, nullptr, true), "Empty complex set failed");
    engine.Eval("assert(isempty(zempty));");
}

void testDirectGetters(MatlabInterface& engine) {
    engine.Eval("g = [1 2 3; 4 5 6];");
    double colMajor[6] = {}, rowMajor[6] = {};
    check(!engine.GetEngineRealMatrix("g", 2, 3, colMajor, true), "Column-major get failed");
    check(!engine.GetEngineRealMatrix("g", 2, 3, rowMajor, false), "Row-major get failed");
    const double expectedCol[] = {1, 4, 2, 5, 3, 6}, expectedRow[] = {1, 2, 3, 4, 5, 6};
    check(std::equal(colMajor, colMajor + 6, expectedCol), "Column-major destination layout");
    check(std::equal(rowMajor, rowMajor + 6, expectedRow), "Row-major destination layout");

    unsigned m = 7, n = 7;
    std::vector<double> vec{9.0};
    check(!engine.GetEngineRealMatrix("g", m, n, vec, false) && m == 2 && n == 3 &&
          vec == std::vector<double>(expectedRow, expectedRow + 6), "Row-major vector get");
    check(!engine.GetEngineRealMatrix("g", m, n, vec, true) &&
          vec == std::vector<double>(expectedCol, expectedCol + 6), "Column-major vector get");

    // Integer, single and logical classes convert through double.
    engine.Eval("i32 = int32([-7 2; 3 2147483647]); s = single([0.5 -1.25]); "
                "l = [true false true]; u8 = uint8([255; 0]); i64 = int64([-2^40 5]);");
    double ints[4], singles[2], logicals[3], bytes[2], wide[2];
    check(!engine.GetEngineRealMatrix("i32", 2, 2, ints, false), "int32 get failed");
    check(ints[0] == -7 && ints[1] == 2 && ints[2] == 3 && ints[3] == 2147483647.0, "int32 conversion");
    check(!engine.GetEngineRealMatrix("s", 1, 2, singles, true) && singles[0] == 0.5 && singles[1] == -1.25,
          "single conversion");
    check(!engine.GetEngineRealMatrix("l", 1, 3, logicals, true) &&
          logicals[0] == 1 && logicals[1] == 0 && logicals[2] == 1, "logical conversion");
    check(!engine.GetEngineRealMatrix("u8", 2, 1, bytes, true) && bytes[0] == 255 && bytes[1] == 0,
          "uint8 conversion");
    check(!engine.GetEngineRealMatrix("i64", 1, 2, wide, true) &&
          wide[0] == -1099511627776.0 && wide[1] == 5, "int64 conversion");

    // Complex destinations: complex double/single and real promotion.
    engine.Eval("zc = [1+2i, 3; -4i, 5-6i]; zs = single([1.5-2i, 0.25i]);");
    Complex zCol[4], zRow[4], zSingle[2], promoted[6];
    check(!engine.GetEngineComplexMatrix("zc", 2, 2, zCol, true), "Complex column-major get failed");
    check(!engine.GetEngineComplexMatrix("zc", 2, 2, zRow, false), "Complex row-major get failed");
    check(zCol[0] == Complex(1, 2) && zCol[1] == Complex(0, -4) && zCol[2] == Complex(3, 0) &&
          zCol[3] == Complex(5, -6), "Complex column-major layout");
    check(zRow[0] == Complex(1, 2) && zRow[1] == Complex(3, 0) && zRow[2] == Complex(0, -4) &&
          zRow[3] == Complex(5, -6), "Complex row-major layout");
    check(!engine.GetEngineComplexMatrix("zs", 1, 2, zSingle, true) &&
          zSingle[0] == Complex(1.5, -2) && zSingle[1] == Complex(0, 0.25), "Complex single conversion");
    check(!engine.GetEngineComplexMatrix("g", 2, 3, promoted, false) &&
          promoted[1] == Complex(2, 0) && promoted[5] == Complex(6, 0), "Real-to-complex promotion");
    std::vector<Complex> zvec;
    check(!engine.GetEngineComplexMatrix("zc", m, n, zvec, false) && m == 2 && n == 2 &&
          zvec == std::vector<Complex>(zRow, zRow + 4), "Complex vector get");

    engine.Eval("empty_get = zeros(0, 4);");
    check(!engine.GetEngineRealMatrix("empty_get", m, n, vec, true) && m == 0 && n == 4 && vec.empty(),
          "Empty vector get");
}

void testGetterErrors(MatlabInterface& engine) {
    // Rejected reads never write the caller's destination.
    const double sentinel = -12345.0;
    double dest[8];
    std::fill(dest, dest + 8, sentinel);
    auto untouched = [&] { return std::all_of(dest, dest + 8, [&](double v) { return v == sentinel; }); };

    engine.Eval("g = [1 2 3; 4 5 6]; nd = zeros(2,2,2); z = [1+1i 2]; txt = 'ab'; c = {1, 2};");
    auto error = thrownMessage([&] { engine.GetEngineRealMatrix("g", 3, 2, dest, true); });
    check(contains(error, "GetEngineRealMatrix: expected size 3x2. Got 2x3.") && untouched(),
          "Size mismatch: " + error);
    error = thrownMessage([&] { engine.GetEngineRealMatrix("nd", 2, 2, dest, true); });
    check(contains(error, "Mat_VarRead invalid dense matrix for nd") && untouched(), "Rank-3 read: " + error);
    // Decode failures take precedence over size checks, as before.
    error = thrownMessage([&] { engine.GetEngineRealMatrix("z", 9, 9, dest, true); });
    check(contains(error, "Failed to decode real matrix data for z") && untouched(), "Complex into real: " + error);
    error = thrownMessage([&] { engine.GetEngineRealMatrix("c", 1, 2, dest, true); });
    check(!error.empty() && untouched(), "Cell array was accepted");
    Complex zdest[4] = {{sentinel, sentinel}, {sentinel, sentinel}, {sentinel, sentinel}, {sentinel, sentinel}};
    error = thrownMessage([&] { engine.GetEngineComplexMatrix("z", 2, 2, zdest, true); });
    check(contains(error, "GetEngineComplexMatrix: expected size 2x2. Got 1x2.") &&
          std::all_of(zdest, zdest + 4, [&](Complex v) { return v == Complex(sentinel, sentinel); }),
          "Complex size mismatch: " + error);
    error = thrownMessage([&] { engine.GetEngineComplexMatrix("nd", 2, 2, zdest, true); });
    check(contains(error, "Mat_VarRead invalid complex matrix for nd"), "Complex rank-3 read: " + error);
    error = thrownMessage([&] { engine.GetEngineComplexMatrix("txt", 1, 2, zdest, true); });
    check(contains(error, "Failed to decode complex matrix data for txt"), "Char into complex: " + error);

    unsigned m = 5, n = 5;
    std::vector<double> vec{1.0, 2.0};
    error = thrownMessage([&] { engine.GetEngineRealMatrix("z", m, n, vec, true); });
    check(contains(error, "Failed to decode real matrix data for z") && m == 0 && n == 0 && vec.empty(),
          "Vector decode failure state: " + error);
    error = thrownMessage([&] { engine.GetEngineRealMatrix("missing_variable", 1, 1, dest, true); });
    check(contains(error, "could not be saved from Octave") && untouched(), "Missing variable: " + error);
}
}

int main(int argc, char** argv) try {
#ifdef BCP_SKIP_SCRIPT_TEST
    std::cout << "Octave runtime unavailable; transport test not run\n";
    return 77;
#endif
    check(argc == 2, "Expected a work directory");
    const fs::path tmp = fs::absolute(argv[1]);
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    // The engine creates its private transfer directory under TMPDIR.
    check(setenv("TMPDIR", tmp.c_str(), 1) == 0, "Cannot set TMPDIR");
    auto& engine = MatlabInterface::GetEngine();

    testRealSetters(engine);
    testComplexSetters(engine);
    testDirectGetters(engine);
    testGetterErrors(engine);
    check(transferFileCount(tmp) == 0, "MAT transfer files were left behind");
    std::cout << "Octave transport tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
