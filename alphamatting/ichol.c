#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>

double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

void _backsub_L_csc_inplace(const double *data, const int *indices, const int *indptr, double *x, int n){
    for (int j = 0; j < n; j++){
        int k = indptr[j];
        double L_jj = data[k];
        double temp = x[j] / L_jj;

        x[j] = temp;

        for (int k = indptr[j] + 1; k < indptr[j + 1]; k++){
            int i = indices[k];
            double L_ij = data[k];

            x[i] -= L_ij * temp;
        }
    }
}

void _backsub_LT_csc_inplace(const double *data, const int *indices, const int *indptr, double *x, int n){
    for (int i = n - 1; i >= 0; i--){
        double s = x[i];

        for (int k = indptr[i] + 1; k < indptr[i + 1]; k++){
            int j = indices[k];
            double L_ji = data[k];
            s -= L_ji * x[j];
        }
        int k = indptr[i];
        double L_ii = data[k];

        x[i] = s / L_ii;
    }
}

void mergesort(int *x, size_t n, int *tmp){
    if (n <= 10){
        for (size_t i = 1; i < n; i++){
            int value = x[i];
            size_t j;
            for (j = i; j > 0 && x[j - 1] > value; j--) x[j] = x[j - 1];
            x[j] = value;
        }
    }else{
        size_t m = n / 2, i = 0, j = m, c = 0;
        mergesort(x, m, tmp);
        mergesort(x + m, n - m, tmp);
        while (i < m && j < n) tmp[c++] = x[i] < x[j] ? x[i++] : x[j++];
        while (i < m) tmp[c++] = x[i++];
        for (size_t k = 0; k < c; k++) x[k] = tmp[k];
    }
}

int _ichol(
    int n,
    const double *Av,
    const int *Ar,
    const int *Ap,
    double *Lv,
    int *Lr,
    int *Lp,
    double droptol,
    int max_nnz
){
    // Data of row j
    int *nonzero_in_row = malloc(n * sizeof(*nonzero_in_row));
    int *next_idx = malloc(n * sizeof(*next_idx));

    // Data of column j
    double *column = malloc(n * sizeof(*column));
    char *used = malloc(n * sizeof(*used));
    int *indices = malloc(n * sizeof(*indices));

    int *tmp = malloc(n * sizeof(*tmp));

    for (int i = 0; i < n; i++){
        nonzero_in_row[i] = -1;
        used[i] = 0;
    }

    int nnz = 0;
    Lp[0] = 0;
    // For each column j of matrix L
    for (int j = 0; j < n; j++){
        // Read column j from matrix A
        int n_indices = 0;
        double threshold = 0.0;
        for (int idx = Ap[j]; idx < Ap[j + 1]; idx++){
            int i = Ar[idx];
            double A_ij = Av[idx];

            // Only consider lower triangular part of A (including diagonal)
            if (i >= j){
                threshold += fabs(A_ij);

                // Activate column element if it is not in use yet
                if (!used[i]){
                    column[i] = 0.0;
                    used[i] = 1;
                    indices[n_indices] = i;
                    n_indices += 1;
                }

                column[i] += A_ij;
            }
        }

        assert(used[j]);

        threshold *= droptol;

        // Compute new values for column j using nonzero values L_jk of row j
        int k = nonzero_in_row[j];
        while (k != -1){
            int idx_start = next_idx[k];
            assert(Lr[idx_start] == j);
            double L_jk = Lv[idx_start];

            // Using nonzero values L_ik of column k
            for (int idx = idx_start; idx < Lp[k + 1]; idx++){
                int i = Lr[idx];
                double L_ik = Lv[idx];

                // Activate column element if it is not in use yet
                if (!used[i]){
                    used[i] = 1;
                    column[i] = 0.0;
                    indices[n_indices] = i;
                    n_indices += 1;
                }

                column[i] -= L_ik * L_jk;
            }

            // Advance to next non-zero element in column k if it exists
            idx_start += 1;
            int k_next = nonzero_in_row[k];
            // Update start of next row
            if (idx_start < Lp[k + 1]){
                next_idx[k] = idx_start;
                int i = Lr[idx_start];

                nonzero_in_row[k] = nonzero_in_row[i];
                nonzero_in_row[i] = k;
            }
            k = k_next;
        }

        if (nnz + n_indices > max_nnz){
            nnz = -2;
            goto ichol_cleanup;
        }

        double diagonal_element = column[j];

        // If not positive definite
        if (diagonal_element <= 0.0){
            nnz = -1;
            goto ichol_cleanup;
        }

        diagonal_element = sqrt(diagonal_element);

        column[j] = diagonal_element;

        // Write diagonal element into matrix L
        Lv[nnz] = diagonal_element;
        Lr[nnz] = j;
        nnz++;
        next_idx[j] = nnz;

        // Output indices must be sorted
        mergesort(indices, n_indices, tmp);

        // Write column j into matrix L
        int first = 1;
        for (int idx = 0; idx < n_indices; idx++){
            int i = indices[idx];

            if (i != j){
                double L_ij = column[i];

                // Drop small values
                if (i != j && fabs(L_ij) >= threshold){
                    // Next row starts here
                    if (first){
                        first = 0;

                        nonzero_in_row[j] = nonzero_in_row[i];
                        nonzero_in_row[i] = j;
                    }

                    // Write element L_ij into L
                    L_ij = L_ij / diagonal_element;
                    Lv[nnz] = L_ij;
                    Lr[nnz] = i;
                    nnz++;
                }
            }

            // Clear column information
            used[i] = 0;
        }

        // Keep track of number of elements per column
        Lp[j + 1] = nnz;
    }

ichol_cleanup:
    free(nonzero_in_row);
    free(next_idx);
    free(column);
    free(used);
    free(indices);
    free(tmp);

    return nnz;
}

