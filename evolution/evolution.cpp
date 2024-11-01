//
// Created by polya on 10/13/24.
//

#include "evolution.hpp"


double evolution::f1(int n1)
{ double x1TmpSquared=x1ValsAllSquared[n1];
    double x1Tmp=x1ValsAll[n1];

    double valTmp = std::exp(-0.5 * omegac * x1TmpSquared)
                    * std::hermite(this->j1H, std::sqrt(omegac) * x1Tmp);


    return valTmp;

}


double evolution::f2(int n2)
{
    double x2TmpSquared=x2ValsAllSquared[n2];
    double x2Tmp=x2ValsAll[n2];


    //    double valTmp=std::exp(-0.5 * omegam*std::exp(-2.0*r) * x2TmpSquared)
    //                  *std::hermite(this->jH2,std::sqrt(omegam*std::exp(-2.0*r))*x2Tmp);
    double valTmp=std::exp(-0.5*omegam*x2TmpSquared)
                  *std::hermite(this->j2H,std::sqrt(omegam)*x2Tmp);

    return valTmp;
}


void evolution::init_psi0()
{
    arma::cx_dcolvec vec1(N1);
    arma::cx_drowvec vec2(N2);
    for(int n1=0;n1<N1;n1++){
        vec1(n1)= f1(n1);
    }
    for(int n2=0;n2<N2;n2++){
        vec2(n2)= f2(n2);
    }
    this->psi0=arma::kron(vec1,vec2);
    std::complex<double> nm(arma::norm(psi0,"fro"),0);
    psi0/=nm;

    // std::cout<<"norm="<<arma::norm(psi0,"fro")<<std::endl;

}

double evolution::rho(const double &x1)
{
double val=omegac*std::pow(x1,2.0)-0.5;

    return val;


}


std::complex<double> evolution::A(const double& x1, const double& x2, const double & tau)
{
double rhoVal=this->rho(x1);
    double rhoVal_squared=std::pow(rhoVal,2.0);

    std::complex<double> part0=1i*this->P1(rhoVal)*tau;

    std::complex<double> part1=1i*F2*rhoVal*x2*std::cos(omegap*tau);

    std::complex<double> part2=1i*F3*rhoVal*x2*std::sin(omegap*tau);

    std::complex<double>part3=1i*F4*rhoVal_squared*std::cos(2*omegap*tau);

    std::complex<double> part4=1i*F5*rhoVal_squared*std::sin(2.0*omegap*tau);

    std::complex<double> part5=1i*F6*rhoVal_squared;

    std::complex<double> part6=1i*F7*std::pow(x2,2.0);

    std::complex<double> part7(0.5*lmd*std::sin(theta)*tau,0);

    std::complex<double> A_Val=part0+part1+part2+part3
               + part4+part5+part6+part7;

    return A_Val;


}


double evolution::P1(const double & rhoVal)
{
double val=0.25*omegac+0.5*Deltam-0.5*omegac*rhoVal+(2.0*omegap-mu)/(2.0*D)*std::pow(g0*rhoVal,2.0);

    return val;


}


std::complex<double> evolution::B(const double& x1, const double& x2, const double & tau)
{

    double rhoVal=this->rho(x1);
    double rhoVal_squared=std::pow(rhoVal,2.0);

    double expVal=std::exp(lmd*std::sin(theta)*tau);

    double expVal_squared=std::exp(2.0*lmd*std::sin(theta)*tau);

    std::complex<double> part0=1i*R1*rhoVal_squared;

    std::complex<double> part1=1i*R2*rhoVal*x2*expVal;

    std::complex<double> part2=1i*R3*rhoVal_squared*std::sin(omegap*tau)*expVal;

    std::complex<double> part3=1i*R4*rhoVal_squared*std::cos(omegap*tau)*expVal;

    std::complex<double> part4=1i*(R5*std::pow(x2,2.0)+R6*rhoVal_squared)*expVal_squared;


    std::complex<double> part5=1i*R7*rhoVal*x2*std::sin(omegap*tau)*expVal_squared;

    std::complex<double> part6=1i*R8*rhoVal*x2*std::cos(omegap*tau)*expVal_squared;

    std::complex<double> part7=1i*R9*rhoVal_squared*std::cos(2.0*omegap*tau)*expVal_squared;

    std::complex<double> part8=1i*R10*rhoVal_squared*std::sin(2.0*omegap*tau)*expVal_squared;

    std::complex<double> B_Val=part0+part1+part2+part3
                                +part4+part5+part6+part7+part8;

    return B_Val;

}