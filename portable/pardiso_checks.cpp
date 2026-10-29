#include "PardisoAPI.h"
#include <cmath>

extern "C" {
void bcp_pardiso_chkmatrix(int*, int* n, double* a, int* ia, int* ja, int* error) {
    *error = 0;
    if (*n < 1 || ia[0] != 1) { *error = -1; return; }
    for (int i = 0; i < *n; ++i) {
        if (ia[i + 1] < ia[i]) { *error = -1; return; }
        for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k)
            if (ja[k] < 1 || ja[k] > *n || !std::isfinite(a[k]) ||
                (k > ia[i] - 1 && ja[k] <= ja[k - 1])) *error = -1;
    }
}
void bcp_pardiso_chkvec(int* n, int* nrhs, double* b, int* error) {
    *error = 0;
    for (int i = 0; i < *n * *nrhs; ++i) if (!std::isfinite(b[i])) *error = -1;
}
void bcp_pardiso_printstats(int*, int*, double*, int*, int*, int*, double*, int* error) { *error = 0; }
}
