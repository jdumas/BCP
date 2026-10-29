#!/usr/bin/env python3
"""Persistent E/G/H worker. JSON lines frame binary column-major double arrays.

Complex payloads contain the real block followed by the imaginary block.
Only prepare uploads Jacobians; evaluate sends x and downloads results. Stdout
is protocol-only, stderr is reserved for diagnostics. No sockets or MAT files.
"""
import json
import os
import sys

import numpy as np

from torch_energy import TorchEnergy


def read_exact(stream, count):
    chunks = bytearray(count)
    view = memoryview(chunks)
    offset = 0
    while offset < count:
        size = stream.readinto(view[offset:])
        if not size:
            raise EOFError("Truncated tensor-worker request")
        offset += size
    return chunks


def read_arrays(stream, descriptors):
    arrays = {}
    for item in descriptors:
        rows, cols = int(item["rows"]), int(item["cols"])
        if rows < 0 or cols < 0 or item["name"] in arrays:
            raise ValueError("Invalid array descriptor")
        count = rows * cols
        complex_data = bool(item["complex"])
        data = np.frombuffer(read_exact(stream, count * 8 * (2 if complex_data else 1)), dtype="<f8")
        if complex_data:
            data = data[:count] + 1j * data[count:]
        arrays[item["name"]] = data.reshape((rows, cols), order="F")
    return arrays


def write_response(stream, arrays, meta):
    descriptors, payloads = [], []
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.ndim != 2:
            raise ValueError(f"Worker result {name} must be a matrix")
        complex_data = np.iscomplexobj(value)
        descriptors.append(dict(name=name, rows=value.shape[0], cols=value.shape[1], complex=complex_data))
        payloads.append(np.asarray(value.real, dtype="<f8").tobytes(order="F"))
        if complex_data:
            payloads.append(np.asarray(value.imag, dtype="<f8").tobytes(order="F"))
    stream.write((json.dumps(dict(ok=True, arrays=descriptors, meta=meta)) + "\n").encode("ascii"))
    for payload in payloads:
        stream.write(payload)
    stream.flush()


def main():
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    evaluator = None
    stats = dict(pid=os.getpid(), prepare_count=0, evaluate_count=0, matrix_upload_bytes=0)
    while True:
        line = source.readline(65537)
        if not line:
            return
        if len(line) > 65536 or not line.endswith(b"\n"):
            raise ValueError("Invalid tensor-worker header")
        try:
            request = json.loads(line)
            arrays = read_arrays(source, request.get("arrays", []))
            operation = request["op"]
            result = {}
            if operation == "start":
                evaluator = None
                evaluator = TorchEnergy(device=request["device"])
                stats["device"] = str(evaluator.device)
            elif evaluator is None:
                raise RuntimeError("Tensor worker has not been started")
            elif operation == "prepare":
                evaluator.prepare(arrays["J_fz"], arrays["J_fbz"], arrays["area"],
                                  arrays["fixed_indices"], arrays["fixed_values"],
                                  hessian_indices=arrays.get("hessian_indices"))
                stats["prepare_count"] += 1
                stats["matrix_upload_bytes"] += sum(a.nbytes for a in arrays.values())
            elif operation == "evaluate":
                result = evaluator.evaluate(arrays["x"], spd=bool(request["spd"]),
                                            exclude_flips=bool(request["exclude_flips"]))
                stats["evaluate_count"] += 1
            elif operation != "stats":
                raise ValueError(f"Unknown tensor-worker operation: {operation}")
            write_response(sink, result, stats)
        except BrokenPipeError:
            return
        except Exception as error:
            try:
                sink.write((json.dumps(dict(ok=False, error=f"{type(error).__name__}: {error}")) + "\n").encode("ascii", errors="backslashreplace"))
                sink.flush()
            except BrokenPipeError:
                return


if __name__ == "__main__":
    main()
