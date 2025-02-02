//
// Created by polya on 10/13/24.
//

#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP
#include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/json.hpp>
#include <cmath>
#include <complex>

#include <cstdio>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
const auto PI=M_PI;
using namespace std::complex_literals;
namespace fs = boost::filesystem;
//This subroutine computes evolution using operator splitting
//one step is exact solution of quasi-linear pde

class evolution
{
public:
    evolution(const std::string &cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty()) {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            //read j1H
            if (paramCounter == 0)
            {
                iss>>j1H;
                if (j1H<0)
                {
                    std::cerr << "j1H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }
            //end reading j1H

            //read j2H
            if (paramCounter == 1)
            {
                iss>>j2H;
                if (j2H<0)
                {
                    std::cerr << "j2H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;

            }
            //end reading j2H

            //read g0
            if (paramCounter == 2)
            {
                iss>>g0;
                paramCounter++;
                continue;
            }
            //end reading g0

            //read omegam
            if(paramCounter == 3)
            {
                iss>>omegam;
                paramCounter++;
                continue;
            }//end reading omegam

            //read omegap
            if(paramCounter == 4)
            {
                iss>>omegap;
                paramCounter++;
                continue;
            }
            //end reading omegap

            //read omegac
            if(paramCounter == 5)
            {
                iss>>omegac;
                paramCounter++;
                continue;
            }
            //end reading omegac

            //read er
            if(paramCounter == 6)
            {
                iss>>er;
                if(er<=0)
                {
                    std::cerr << "er must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }
            //end reading er

            //read thetaCoef
            if(paramCounter == 7)
            {
                iss>>thetaCoef;
                paramCounter++;
                continue;
            }
            //end reading thetaCoef


        }//end while

        //print parameters
        std::cout << std::setprecision(15);
        std::cout<<"j1H="<<j1H<<", j2H="<<j2H<<", g0="<<g0
        <<", omegam="<<omegam<<", omegap="<<omegap<<", omegac="<<omegac
        <<", er="<<er<<", thetaCoef="<<thetaCoef<<std::endl;


        this->L1=5;
        this->L2=8;
        this->r=std::log(er);
        this->theta=thetaCoef*PI;
        this->Deltam=omegam-omegap;
        std::cout<<"Deltam="<<Deltam<<std::endl;
        this->e2r=std::pow(er,2.0);

        this->lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam;
        std::cout<<"lambda="<<lmd<<std::endl;
        double height1=0.5;
        double width1=std::pow(-2.0*std::log(height1)/omegac,0.5);
        double minGrid1=width1/10.0;
        this->N2=600;

        this->N1=static_cast<int>(std::ceil(L1*2.0/minGrid1));
        if(N1%2==1)
        {
            N1+=1;
        }
        std::cout<<"L1="<<L1<<", L2="<<L2<<std::endl;
        std::cout<<"N1="<<N1<<std::endl;
        std::cout<<"N2="<<N2<<std::endl;

        dx1=2.0*L1/static_cast<double>(N1);
        dx2=2.0*L2/static_cast<double>(N2);
        std::cout<<"dx1="<<dx1<<std::endl;
        std::cout<<"dx2="<<dx2<<std::endl;



        for (int n1 =0;n1<N1;n1++){
            this->x1ValsAll.push_back(-L1+dx1*n1);
        }
        for (int n2=0;n2<N2;n2++){
            this->x2ValsAll.push_back(-L2+dx2*n2);
        }
        for(const auto& val: x1ValsAll){
            x1ValsAllSquared.push_back(std::pow(val,2));
        }
        for(const auto &val:x2ValsAll){
            x2ValsAllSquared.push_back(std::pow(val,2));
        }
        for(int n1=0;n1<static_cast<int>(N1/2);n1++){
            k1ValsAll_fft.push_back(2*PI*static_cast<double >(n1)/(2.0*L1));
        }
        for(int n1=static_cast<int>(N1/2);n1<N1;n1++){
            k1ValsAll_fft.push_back(2*PI*static_cast<double >(n1-N1)/(2.0*L1));
        }

        for(int n1=0;n1<N1;n1++)
        {
            k1ValsAll_interpolation.push_back(2*PI*static_cast<double>(n1)/(2.0*L1));

        }
        for(const auto&val: k1ValsAll_fft){
            k1ValsAllSquared_fft.push_back(std::pow(val,2));
        }
        for(int n2=0;n2<static_cast<int>(N2/2);n2++){
            k2ValsAll_fft.push_back(2*PI*static_cast<double >(n2)/(2.0*L2));
        }
        for(int n2=static_cast<int >(N2/2);n2<N2;n2++){
            k2ValsAll_fft.push_back(2*PI*static_cast<double >(n2-N2)/(2.0*L2));
        }

        for(int n2=0;n2<N2;n2++)
        {
            k2ValsAll_interpolation.push_back(2*PI*static_cast<double>(n2)/(2.0*L2));
        }


        for(const auto &val:k2ValsAll_fft){
            k2ValsAllSquared_fft.push_back(std::pow(val,2));
        }

        //initialize parameters for A

        D=std::pow(lmd*std::sin(theta),2.0)+std::pow(omegap,2.0);
        mu=lmd*std::cos(theta)+Deltam;
        std::cout<<"D="<<D<<std::endl;
        std::cout<<"mu="<<mu<<std::endl;
        this->F2=g0*std::sqrt(2.0*omegam)*(2.0*std::pow(lmd*std::sin(theta),2.0)+omegap*mu)/(2*D*lmd*sin(theta));

        this->F3=g0*std::sqrt(2.0*omegam)/D*(0.5*mu-omegap);

        this->F4=std::pow(g0,2.0)*(2.0*D*std::pow(lmd*std::sin(theta),2.0)+mu*omegap*std::pow(lmd*std::sin(theta),2.0)+mu*std::pow(omegap,3.0))/(4.0*std::pow(D,2.0)*lmd*omegap*std::sin(theta));

        this->F5=std::pow(g0,2.0)*
            (2.0*omegap*D+mu*std::pow(lmd*std::sin(theta),2.0)-4.0*omegap*std::pow(lmd*std::sin(theta),2.0)+mu*std::pow(omegap,2.0)-4.0*std::pow(omegap,3.0))
        /(4.0*omegap*std::pow(D,2.0));

        this->F6=std::pow(g0,2.0)*
            (8.0*omegap*std::pow(lmd*std::sin(theta),2.0)-4.0*mu*std::pow(lmd*std::sin(theta),2.0)+D*mu)
            /(4*lmd*std::sin(theta)*std::pow(D,2.0));

        this->F7=omegam*mu/(4*lmd*std::sin(theta));

        std::cout<<"F2="<<F2<<std::endl;
        std::cout<<"F3="<<F3<<std::endl;
        std::cout<<"F4="<<F4<<std::endl;
        std::cout<<"F5="<<F5<<std::endl;
        std::cout<<"F6="<<F6<<std::endl;
        std::cout<<"F7="<<F7<<std::endl;
        //end initializing parameters for A

        //initialize parameters for B
        this->R1=std::pow(g0,2.0)*lmd*std::sin(theta)/(2.0*omegap)
                *(D-omegap*mu)/std::pow(D,2.0);

        this->R2=g0*lmd*std::sin(theta)/D*std::sqrt(2.0*omegam);

        this->R3=-2.0*std::pow(g0*lmd*std::sin(theta),2.0)/std::pow(D,2.0);

        this->R4=2.0*std::pow(g0,2.0)*omegap*lmd*std::sin(theta)/std::pow(D,2.0);

        this->R5=omegam/(4.0*lmd*std::sin(theta))*mu;

        this->R6=mu*std::pow(g0,2.0)/(4.0*lmd*std::sin(theta)*D);

        this->R7=-mu*g0/(2.0*D)*std::sqrt(2.0*omegam);

        this->R8=mu*g0*omegap/(2.0*lmd*std::sin(theta)*D)*std::sqrt(2.0*omegam);

        this->R9=mu*std::pow(g0,2.0)/(4.0*lmd*std::sin(theta)*std::pow(D,2.0))*(std::pow(omegap,2.0)-std::pow(lmd*std::sin(theta),2.0));

        this->R10=-mu*omegap*std::pow(g0,2.0)/(2*std::pow(D,2.0));

        std::cout<<"R1="<<R1<<std::endl;
        std::cout<<"R2="<<R2<<std::endl;
        std::cout<<"R3="<<R3<<std::endl;

        std::cout<<"R4="<<R4<<std::endl;
        std::cout<<"R5="<<R5<<std::endl;
        std::cout<<"R6="<<R6<<std::endl;

        std::cout<<"R7="<<R7<<std::endl;
        std::cout<<"R8="<<R8<<std::endl;
        std::cout<<"R9="<<R9<<std::endl;

        std::cout<<"R10="<<R10<<std::endl;

        //+-1, for d_arma
        this->multiplier_of_d_arma=std::vector<std::complex<double>>(N2);

        for (int j=0;j<N2;j++)
        {
            multiplier_of_d_arma[j]=std::exp(std::complex<double>(0, PI * j));
        }
        this->one_over_N2=std::complex<double>(1.0/static_cast<double>(N2),0);

        //end initializing parameters for B

        //matrices
        this->construct_S_mat_spatial();

        //pointers
        this->d_ptr=new std::complex<double>[N1*N2];
        this->Phi=new std::complex<double>[N1*N2];
        this->D_widehat=new std::complex<double>[N1*N2];
        this->I_widehat=new std::complex<double>[N1*N2];
        this->J_widehat=new std::complex<double>[N1*N2];
        //arma matrices
        this->d_arma=arma::cx_dmat(N1,N2);
        this->c_arma=arma::cx_dmat(N1,N2);
        this->F_widehat_arma=arma::cx_dmat(N1,N2);
        this->G_widehat_arma=arma::cx_dmat(N1,N2);

        //plans

        //plan Phi to d_ptr, column fft
       this->M1_Phi=N2;
       this->M2_phi=N1;
       int rank_Phi_2_d_ptr=1;

       int n_Phi[]={M1_Phi};
       int how_many_Phi=M2_phi;
       int istride_Phi=M2_phi, ostride_Phi=M2_phi;
       int idist_Phi=1, odist_Phi=1;
        plan_col_Phi_2_d_ptr=fftw_plan_many_dft(
                rank_Phi_2_d_ptr,n_Phi,how_many_Phi,
                reinterpret_cast < fftw_complex * >(Phi), NULL,
                istride_Phi,idist_Phi,
                reinterpret_cast < fftw_complex * >(d_ptr),NULL,
                ostride_Phi,odist_Phi,
                FFTW_FORWARD, FFTW_MEASURE
                );

       //end plan Phi to d_ptr, column fft
        // double x1Tmp=0.1;
        // double x2Tmp=0.2;
        // double tauTmp=0.04;


        // arma::dmat S2Tmp=construct_S_mat(tauTmp);

        // int n1=10;
        // int n2=134;
        // std::cout<<"S2Tmp(n1,n2)="<<S2Tmp(n1,n2)<<std::endl;
        // std::complex<double> A_val_Tmp=A(x1Tmp,x2Tmp,tauTmp);
        //      std::cout<<"A_val_Tmp="<<A_val_Tmp<<std::endl;

        // std::complex<double> B_val_tmp=B(x1Tmp,x2Tmp,tauTmp);
        // std::cout<<"B_val_tmp="<<B_val_tmp<<std::endl;

        this->plan_2d_fft_Phi_2_D_widehat=fftw_plan_dft_2d(N2,N1,
            reinterpret_cast<fftw_complex*>(Phi),
            reinterpret_cast<fftw_complex*>(D_widehat),
           FFTW_FORWARD, FFTW_MEASURE);

        this->plan_2d_ifft_I_widehat_2_J_widehat=fftw_plan_dft_2d(N2,N1,
            reinterpret_cast<fftw_complex*>(I_widehat),
            reinterpret_cast<fftw_complex*>(J_widehat),FFTW_BACKWARD,FFTW_MEASURE );
this->alpha=0.5;
        this->beta=1.0;
this->gamma13=1.0/(2.0-std::pow(2.0,1.0/3.0));
        this->gamma23=-std::pow(2.0,1.0/3.0)/(2.0-std::pow(2.0,1.0/3.0));
        this->gamma15=1.0/(2.0-std::pow(2.0,1.0/5.0));
        this->gamma25=-std::pow(2.0,1.0/5.0)/(2.0-std::pow(2.0,1.0/5.0));
        std::cout<<"alpha="<<alpha<<std::endl;
        std::cout<<"beta="<<beta<<std::endl;
        std::cout<<"gamma13="<<gamma13<<std::endl;
        std::cout<<"gamma23="<<gamma23<<std::endl;
        std::cout<<"gamma15="<<gamma15<<std::endl;
        std::cout<<"gamma25="<<gamma25<<std::endl;
        //construct tree1
        this->tree1_level1={gamma13,gamma23,gamma13};
        this->tree1_level2={alpha,beta,alpha};
        for (int i=0;i<3;i++)
        {
            for (int j=0;j<3;j++)
            {
                std::vector<double>tmp={tree1_level2[j],tree1_level1[i],gamma15};
                tree1.push_back(tmp);
            }//end j
        }//end i
        //construct tree2
        tree2_level1={gamma13,gamma23,gamma13};
        tree2_level2={alpha,beta,alpha};
        for (int i=0;i<3;i++)
        {
            for (int j=0;j<3;j++)
            {
                std::vector<double>tmp={tree2_level2[j],tree2_level1[i],gamma25};
                tree2.push_back(tmp);
            }
        }

        //construct tree2
        tree3_level1={gamma13,gamma23,gamma13};
        tree3_level2={alpha,beta,alpha};
        for (int i=0;i<3;i++)
        {
            for (int j=0;j<3;j++)
            {
                std::vector<double>tmp={tree3_level2[j],tree3_level1[i],gamma15};
                tree3.push_back(tmp);
            }//end j
        }//end i
        // std::cout<<"tree3:\n";
        // for (int i =0;i<9;i++)
        // {
        //     printVec(tree3[i]);
        // }

        this->U1_inds={0,2,3,5,6,8};
        this->U2_inds={1,4,7};
        this->tTot=5.0;
        this->Q=static_cast<int>(1e6);
        this->dt=tTot/static_cast<double>(Q);
        std::cout<<"tTot="<<tTot<<std::endl;
        std::cout<<"Q="<<Q<<std::endl;
        std::cout<<"dt="<<dt<<std::endl;
    }//end constructor

 ~ evolution()
    {
        delete[] d_ptr;
        delete[] Phi;
        delete [] D_widehat;
        delete [] I_widehat;
        delete [] J_widehat;

        fftw_destroy_plan(plan_col_Phi_2_d_ptr);
        fftw_destroy_plan(plan_2d_fft_Phi_2_D_widehat);
        fftw_destroy_plan(plan_2d_ifft_I_widehat_2_J_widehat);
    }
public:
    ///
    ///initialize A,B,S2,V in tree1
    void init_tree1_mats();


    void init_tree2_mats();
    void init_tree3_mats();
    arma::cx_dmat construct_V_mat(const double &Delta_t);
    ///
    /// @param Psi_arma wavefunction in cx_dmat
    /// @param Delta_t time step
    /// this function computes evolution in U2
    void step_U2(arma::cx_dmat & Psi_arma, const double &Delta_t);

    ///
    /// @param Delta_t time step
    /// @param n1
    /// @param n2
    /// @return element of V
    std::complex<double> V_elem(const double &Delta_t, const int &n1, const int &n2);

    std::complex<double> * Phi_2_c_arma();
    ///
    /// @param psi wavefunction matrix
    /// @return raw data pointer, column major order, in the note, the content in pointer is Phi
    std::complex<double> * cx_dmat_2_complex_ptr(const arma::cx_dmat& psi);


    arma::dmat construct_S_mat( const double &tau);

    void construct_S_mat_spatial();

    arma::cx_dmat construct_A_mat(const double & tau);


    arma::cx_dmat construct_B_mat(const double & tau);



    std::complex<double> B(const double& x1, const double& x2, const double & tau);

    std::complex<double> A(const double& x1, const double& x2, const double & tau);


    double P1(const double & rhoVal);

    double rho(const double &x1);
    ///
    /// @param n1 index of x1
    /// @return wavefunction of photon at n1
    double f1(int n1);

    ///
    /// @param n2 index of x2
    /// @return wavefunction of phonon at n2
    double f2(int n2);

    void init();


    void init_psi0();



    template<class T>
   static void printVec(const std::vector <T> &vec) {
        for (int i = 0; i < vec.size() - 1; i++) {
            std::cout << vec[i] << ",";
        }
        std::cout << vec[vec.size() - 1] << std::endl;
    }
public:
    int j1H;
    int j2H;
    double g0;
    double omegam;
    double omegap;
    double omegac ;
    double er ;
    double thetaCoef ;
    int groupNum ;
    int rowNum ;
    double theta;
    double lmd;
    double Deltam;
    double r;
    double e2r;

    // double E1;
    // double E2;


    int N1;//must be even
    int N2;//must be even

    double L1;
    double L2;
    double dx1;

    double dx2;

    double dtEst;
    double tTot;
    double dt;
    int Q;
    std::vector<double> x1ValsAll;
    std::vector<double> x2ValsAll;
    std::vector<double> k1ValsAll_fft;
    std::vector<double> k2ValsAll_fft;
    std::vector<double> x1ValsAllSquared;
    std::vector<double> x2ValsAllSquared;
    std::vector<double> k1ValsAllSquared_fft;
    std::vector<double> k2ValsAllSquared_fft;

    std::vector<double> k1ValsAll_interpolation;
    std::vector<double>k2ValsAll_interpolation;


   double alpha;
    double beta;
    double gamma13;
    double gamma23;
    double gamma15;
    double gamma25;
    //parameters for A
    double D;
    double mu;
    double F2,F3,F4,F5,F6,F7;

    //parameters for B

    double R1,R2,R3,R4,R5,R6,R7,R8,R9,R10;

    arma::cx_dmat psi0_arma;//armadillo psi0
    std::complex<double> * d_ptr;
    std::complex<double> * Phi;
    arma::cx_dmat d_arma;
    arma::cx_dmat c_arma;


    arma::dmat S2_mat_part1,S2_mat_part2,S2_mat_part3,S2_mat_part4;

    //plan Phi to d_ptr
    fftw_plan plan_col_Phi_2_d_ptr;
    int M1_Phi;
    int M2_phi;

    std::vector<std::complex<double>> multiplier_of_d_arma;
    std::complex<double>one_over_N2;


    //data for U2
    arma::cx_dmat F_widehat_arma,G_widehat_arma;
    std::complex<double> * D_widehat;
  std::complex<double> *  I_widehat;
    std::complex<double> * J_widehat;

    //2d fft
    fftw_plan plan_2d_fft_Phi_2_D_widehat;
    fftw_plan plan_2d_ifft_I_widehat_2_J_widehat;


    std::vector<double >tree1_level1;
    std::vector<double> tree1_level2;
    std::vector<std::vector<double>> tree1;

    std::vector<double >tree2_level1;
    std::vector<double> tree2_level2;
    std::vector<std::vector<double>> tree2;

    std::vector<double >tree3_level1;
    std::vector<double> tree3_level2;
    std::vector<std::vector<double>> tree3;

    std::vector<arma::cx_dmat> tree1_A_mat_all;
    std::vector<arma::cx_dmat>tree1_B_mat_all;
    std::vector<arma::cx_dmat>tree1_S2_mat_all;
    std::vector<arma::cx_dmat>tree1_V_mat_all;

    std::vector<arma::cx_dmat> tree2_A_mat_all;
    std::vector<arma::cx_dmat>tree2_B_mat_all;
    std::vector<arma::cx_dmat>tree2_S2_mat_all;
    std::vector<arma::cx_dmat>tree2_V_mat_all;

    std::vector<arma::cx_dmat> tree3_A_mat_all;
    std::vector<arma::cx_dmat>tree3_B_mat_all;
    std::vector<arma::cx_dmat>tree3_S2_mat_all;
    std::vector<arma::cx_dmat>tree3_V_mat_all;

std::vector<int> U1_inds;
    std::vector<int> U2_inds;

};

#endif //EVOLUTION_HPP
