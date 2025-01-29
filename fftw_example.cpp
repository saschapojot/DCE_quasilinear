#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <iostream>


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

void fft_columns ( std :: complex < double >* input , std :: complex < double >* output , int M1 , int M2 ){
// Define FFT parameters for column - wise FFT
    int rank = 1; // 1 D FFT
    int n [] = { M1 }; // Length of each FFT is M1 ( number of rows )
    int howmany = M2 ; // Number of FFTs is M2 ( number of columns )
    int istride = M2 , ostride = M2 ; // Stride for column - wise FFT
    int idist = 1 , odist = 1; // Distance between start of each column FFT

    // Create FFTW plan for column - wise , out - of - place transform

    fftw_plan plan = fftw_plan_many_dft (
            rank , n , howmany ,
            reinterpret_cast < fftw_complex * >( input ) , NULL ,
            istride , idist ,
            reinterpret_cast < fftw_complex * >( output ) , NULL ,
            ostride , odist ,
            FFTW_FORWARD , FFTW_ESTIMATE
    ) ;

    // Execute the FFT plan
    fftw_execute ( plan ) ;

    // Clean up
    fftw_destroy_plan ( plan ) ;

}

int main()
{
    arma::cx_mat Psi(4, 4); // Initialize with zeros
    // // Fill the matrix with values from 0 to 15
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            Psi(i, j) = std::complex<double>(i * 4 + j, 0.0); // Real part = 0 to 15, Imaginary part = 0
        }
    }
    Psi.print("Psi:");

    // Get the pointer to the internal storage
    std::complex<double>* data_ptr = Psi.memptr();
    // Print the stored values to verify
    // std::cout << "Internal storage (column-major order):\n";
    // for (size_t i = 0; i < 16; ++i) {
    //     std::cout << data_ptr[i] << " ";
    // }
    // std::cout << std::endl;
    std::complex<double> * d_ptr=new std::complex<double>[16];
    fft_columns(data_ptr, d_ptr, 4, 4);
    arma::cx_mat output(d_ptr,4, 4);

    output.print("output:");
    delete [] d_ptr;


    return 0;
}