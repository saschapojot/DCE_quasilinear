#include <fftw3.h>
#include <complex>
#include <iostream>

void fft_rows(std::complex<double>* input, std::complex<double>* output, int N1, int N2) {
    int rank = 1;
    int n[] = {N2};
    int howmany = N1;
    int istride = 1, ostride = 1;
    int idist = N2, odist = N2;

    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL,
        istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL,
        ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(plan);

    // Normalize the output
    for (int i = 0; i < N1 * N2; ++i) {
        output[i] /= N2;
    }

    fftw_destroy_plan(plan);
}

int main() {
    int N1 = 4;  // Number of rows
    int N2 = 8;  // Number of columns (length of FFT)

    std::complex<double>* input = new std::complex<double>[N1 * N2];
    std::complex<double>* output = new std::complex<double>[N1 * N2];

    // Initialize input matrix with example data
    for (int i = 0; i < N1 * N2; ++i) {
        input[i] = std::complex<double>(i, 0);
    }

    // Perform FFT on each row (out-of-place)
    fft_rows(input, output, N1, N2);

    std::cout << "Transformed (FFT) output matrix:" << std::endl;
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            std::cout << output[i * N2 + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] input;
    delete[] output;
    return 0;
}



void fft_columns(std::complex<double>* input, std::complex<double>* output, int N1, int N2) {
    // Define FFT parameters for column-wise FFT
    int rank = 1;                   // 1D FFT
    int n[] = {N1};                 // Length of each FFT is N1 (number of rows)
    int howmany = N2;               // Number of FFTs is N2 (number of columns)
    int istride = N2, ostride = N2; // Stride for column-wise FFT
    int idist = 1, odist = 1;       // Distance between start of each column FFT

    // Create FFTW plan for column-wise, out-of-place transform
    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL,
        istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL,
        ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    // Execute the FFT plan
    fftw_execute(plan);

    // Clean up
    fftw_destroy_plan(plan);
}