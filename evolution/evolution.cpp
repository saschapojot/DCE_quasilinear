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
