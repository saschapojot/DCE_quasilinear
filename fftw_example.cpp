#include <fftw3.h>
#include <complex>
#include <iostream>


void fft_rows(std::complex<double>* input, std::complex<double>* output, int N1, int N2) {
    int rank = 1;                   // 1D FFT
    int n[] = {N2};                 // Length of each 1D FFT
    int howmany = N1;               // Number of 1D FFTs (one per row)
    int istride = 1, ostride = 1;   // Contiguous elements in each row
    int idist = N2, odist = N2;     // Distance between consecutive rows

    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL,
        istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL,
        ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(plan);
    fftw_destroy_plan(plan);
}