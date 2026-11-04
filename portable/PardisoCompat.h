#pragma once
#ifdef BCP_PARDISO_COMPAT
#include "PardisoAPI.h"
#define pardisoinit bcp_pardisoinit
#define pardiso bcp_pardiso
#define pardiso_chkmatrix bcp_pardiso_chkmatrix
#define pardiso_chkvec bcp_pardiso_chkvec
#define pardiso_printstats bcp_pardiso_printstats
#else
// The original Panua ABI, used only when explicitly selected.
extern "C" {
void pardisoinit(void*, int*, int*, int*, double*, int*);
void pardiso(void*, int*, int*, int*, int*, int*, double*, int*, int*, int*,
             int*, int*, int*, double*, double*, int*, double*);
void pardiso_chkmatrix(int*, int*, double*, int*, int*, int*);
void pardiso_chkvec(int*, int*, double*, int*);
void pardiso_printstats(int*, int*, double*, int*, int*, int*, double*, int*);
}
#endif
