#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <iomanip> // For std::setprecision
#include <iostream>
#include <cstring> // For std::memcpy

void fft_rows(std::complex<double>* input, std::complex<double>* output, int M1, int M2) {
    int rank = 1;                   // 1D FFT
    int n[] = {M2};                 // Length of each 1D FFT
    int howmany = M1;               // Number of 1D FFTs (one per row)
    int istride = 1, ostride = 1;   // Contiguous elements in each row
    int idist = M2, odist = M2;     // Distance between consecutive rows

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

void fft_columns(std::complex<double>* input, std::complex<double>* output, int M1, int M2) {
    int rank = 1; // 1D FFT
    int n[] = {M1}; // Length of each FFT is M1 (number of rows)
    int howmany = M2; // Number of FFTs is M2 (number of columns)
    int istride = M2, ostride = M2; // Stride for column-wise FFT
    int idist = 1, odist = 1; // Distance between start of each column FFT

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
void print_complex_matrix(const std::complex<double>* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(2)
                      << "(" << data[i * cols + j].real() << ", " << data[i * cols + j].imag() << ") ";
        }
        std::cout << std::endl;
    }
}
int main() {
    int Nx = 4;
    int Ny = 4;

    // Define input and output arrays using std::complex<double>*
    std::complex<double> *in, *out, *temp;

    // Allocate memory using fftw_malloc
    in = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * Nx * Ny);
    out = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * Nx * Ny);
    temp = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * Nx * Ny); // Temporary buffer

    // Create a 2D FFTW plan for forward transformation
    fftw_plan plan_forward = fftw_plan_dft_2d(Nx, Ny,
                                              reinterpret_cast<fftw_complex*>(in),
                                              reinterpret_cast<fftw_complex*>(out),
                                              FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_backward=fftw_plan_dft_2d(Nx,Ny,
        reinterpret_cast<fftw_complex*>(out),
         reinterpret_cast<fftw_complex*>(temp),
         FFTW_BACKWARD,FFTW_MEASURE);


    // Initialize input data with values from 0 to 15 (row-major order)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            in[i * Ny + j] = std::complex<double>(i * Ny + j, 0.0); // Real part = 0,1,2,...,15
        }
    }

    fftw_execute(plan_forward);
    std::cout<<"==============="<<std::endl;
    std::cout<<"forward:\n";
    print_complex_matrix(out, Nx, Ny);

    fftw_execute(plan_backward);
    // Normalize the inverse FFT result
    double norm = 1.0 / (Nx * Ny);
    for (int i = 0; i < Nx * Ny; ++i) {
        temp[i] *= norm;
    }
    std::cout<<"==============="<<std::endl;
    std::cout << "\nInverse FFT Result (Should Match Input):" << std::endl;
    print_complex_matrix(temp, Nx, Ny);





    // Cleanup: destroy plans and free memory
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);

    delete [] in;
    delete [] out;
    delete [] temp;
    // fftw_free(temp);

    return 0;
}