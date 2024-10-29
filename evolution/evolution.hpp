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
            k1ValsAll_intepolation.push_back(2*PI*static_cast<double>(n1)/(2.0*L1));

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


    }//end constructor

    // H1^{R} part
    double rho(const double &x1);
public:
    ///
    /// @param n1 index of x1
    /// @return wavefunction of photon at n1
    double f1(int n1);

    ///
    /// @param n2 index of x2
    /// @return wavefunction of phonon at n2
    double f2(int n2);


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

    double L1=5;
    double L2=8;
    double dx1;

    double dx2;

    double dtEst;
    std::vector<double> x1ValsAll;
    std::vector<double> x2ValsAll;
    std::vector<double> k1ValsAll_fft;
    std::vector<double> k2ValsAll_fft;
    std::vector<double> x1ValsAllSquared;
    std::vector<double> x2ValsAllSquared;
    std::vector<double> k1ValsAllSquared_fft;
    std::vector<double> k2ValsAllSquared_fft;

    std::vector<double> k1ValsAll_intepolation;
    std::vector<double>k2ValsAll_interpolation;

    arma::cx_dmat psi0;

    //parameters for A
    double D;
    double mu;
    double F2,F3,F4,F5,F6,F7;
};

#endif //EVOLUTION_HPP
