#pragma once
// Minimal documented LP64 declarations for adapter contract tests, NOT an MKL SDK.
using MKL_INT = int;
extern "C" {
void pardisoinit(void*, const MKL_INT*, MKL_INT*);
void pardiso(void*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const MKL_INT*,
             const MKL_INT*, const void*, const MKL_INT*, const MKL_INT*, MKL_INT*,
             const MKL_INT*, MKL_INT*, const MKL_INT*, void*, void*, MKL_INT*);
}
