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

void fft_columns ( std :: complex < double >* input , std :: complex < double >* output , int N1 , int N2 ){
// Define FFT parameters for column - wise FFT
    int rank = 1; // 1 D FFT
    int n [] = { N1 }; // Length of each FFT is N1 ( number of rows )
    int howmany = N2 ; // Number of FFTs is N2 ( number of columns )
    int istride = N2 , ostride = N2 ; // Stride for column - wise FFT
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