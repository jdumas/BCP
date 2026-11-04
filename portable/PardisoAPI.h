#pragma once

// Private compatibility ABI. Never export Intel/Panua's own symbol names.
// The legacy callers retain one-based CSR and int (LP64) indices.
extern "C" {
void bcp_pardisoinit(void*, int*, int*, int*, double*, int*);
void bcp_pardiso(void*, int*, int*, int*, int*, int*, double*, int*, int*, int*,
                 int*, int*, int*, double*, double*, int*, double*);
void bcp_pardiso_chkmatrix(int*, int*, double*, int*, int*, int*);
void bcp_pardiso_chkvec(int*, int*, double*, int*);
void bcp_pardiso_printstats(int*, int*, double*, int*, int*, int*, double*, int*);
}
