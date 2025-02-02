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
    this->psi0_arma=arma::kron(vec1,vec2);
    std::complex<double> nm(arma::norm(psi0_arma,"fro"),0);
    psi0_arma/=nm;

    // std::cout<<"norm="<<arma::norm(psi0_arma,"fro")<<std::endl;

}

double evolution::rho(const double &x1)
{
double val=omegac*std::pow(x1,2.0)-0.5;

    return val;


}
void evolution::init()
{

    this->init_psi0();
    this->construct_S_mat_spatial();

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


arma::dmat evolution::construct_S_mat(const double &tau)
{

    double exp_part= std::exp(lmd*std::sin(theta)*tau);

    double sin_val=std::sin(omegap*tau);

    double cos_val=std::cos(omegap*tau);

    arma::dmat S2_mat=S2_mat_part1*exp_part+S2_mat_part2*sin_val*exp_part
                     + S2_mat_part3*cos_val*exp_part+S2_mat_part4;


    return S2_mat;



}


void evolution::construct_S_mat_spatial()
{
    this->S2_mat_part1=arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part2=arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part3=arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part4=arma::dmat(N1, N2, arma::fill::zeros);

    //begin initializing S2_mat_part1
    for (int n1=0;n1<N1;n1++)
    {
        for (int n2=0;n2<N2;n2++)
        {
            S2_mat_part1(n1,n2)=this->x2ValsAll[n2];

        }//end n2
    }//end n1, end initializing S2_mat_part1

    //begin initializing S2_mat_part2
    for (int n1=0;n1<N1;n1++)
    {
        double x1n1=this->x1ValsAll[n1];
        double rhoTmp=this->rho(x1n1);
        for (int n2=0;n2<N2;n2++)
        {
            S2_mat_part2(n1,n2)=-g0*lmd*std::sin(theta)/D*std::sqrt(2.0/omegam)*rhoTmp;
        }//end n2

    }//end n1, end initializing S2_mat_part2

    //begin initializing S2_mat_part3
    for(int n1=0;n1<N1;n1++)
    {
        double x1n1=this->x1ValsAll[n1];
        double rhoTmp=this->rho(x1n1);
        for (int n2=0;n2<N2;n2++)
        {
            S2_mat_part3(n1,n2)=g0*omegap/D*std::sqrt(2.0/omegam)*rhoTmp;
        }//end n2

    }//end n1, end initializing S2_mat_part3


    //begin initializing S2_mat_part4
    for(int n1=0;n1<N1;n1++)
    {
        double x1n1=this->x1ValsAll[n1];
        double rhoTmp=this->rho(x1n1);
        for(int n2=0;n2<N2;n2++)
        {
            S2_mat_part4(n1,n2)=-g0*omegap/D*std::sqrt(2.0/omegam)*rhoTmp;
        }//end n2
    }//end n1, end initializing S2_mat_part4

}



arma::cx_dmat evolution::construct_A_mat(const double & tau)
{

    arma::cx_dmat A_mat(N1,N2,arma::fill::zeros);

    for (int n1=0;n1<N1;n1++)
    {
        for(int n2=0;n2<N2;n2++)
        {
            double x1n1=this->x1ValsAll[n1];
            double x2n2=this->x2ValsAll[n2];

            A_mat(n1,n2)=this->A(x1n1,x2n2,tau);
        }//end n2
    }//end n1
    return A_mat;
}

arma::cx_dmat evolution::construct_B_mat(const double & tau)
{

    arma::cx_dmat B_mat(N1,N2,arma::fill::zeros);
    for(int n1=0;n1<N1;n1++)
    {
        for(int n2=0;n2<N2;n2++)
        {
            double x1n1=this->x1ValsAll[n1];
            double x2n2=this->x2ValsAll[n2];
            B_mat(n1,n2)=this->B(x1n1,x2n2,tau);
        }//end n2
    }//end n1

    return B_mat;
}


std::complex<double> * evolution::cx_dmat_2_complex_ptr(const arma::cx_dmat& psi){

    return const_cast<std::complex<double>*>(psi.memptr());

}


std::complex<double> * evolution::Phi_2_c_arma(){

    // Phi to d_ptr, column fft
    fftw_execute(plan_col_Phi_2_d_ptr);

    //d_ptr to d_arma
    this->d_arma=arma::cx_dmat(d_ptr,N1,N2,true);

    //multiply jth row of d_arma with multiplier_of_d_arma[j]
    for(int j=0;j<N2;j++)
    {
        d_arma.col(j)*=multiplier_of_d_arma[j];
    }

    c_arma=d_arma*one_over_N2;

}


///
/// @param Psi_arma wavefunction in cx_dmat
/// @param Delta_t time step
/// this function computes evolution in U2
void evolution::step_U2(arma::cx_dmat & Psi_arma, const double &Delta_t)
{

//Psi to Phi
    this->Phi=Psi_arma.memptr();

    //Phi to D_widehat
    fftw_execute(plan_2d_fft_Phi_2_D_widehat);

    //D_widehat to F_widehat_arma
    this->F_widehat_arma=arma::cx_dmat(this->D_widehat,N1,N2,true);



}




///
/// @param Delta_t time step
/// @param n1
/// @param n2
/// @return element of V
std::complex<double> evolution::V_elem(const double &Delta_t, const int &n1, const int &n2)
{
    double tmp=(lmd*std::cos(theta)-Deltam)/(2.0*omegam)*k2ValsAllSquared_fft[n2]-0.5*k1ValsAllSquared_fft[n1];
    tmp*=Delta_t;

    std::complex<double> to_exp=std::complex<double>(0.0,tmp);

    std::complex<double> elem_tmp=std::exp(to_exp);

    return elem_tmp;


}

arma::cx_dmat evolution::construct_V_mat(const double &Delta_t)
{
    arma::cx_dmat V_mat(N1,N2,arma::fill::zeros);
    for (int n1=0;n1<N1;n1++)
    {
        for (int n2=0;n2<N2;n2++)
        {
            V_mat(n1,n2)=this->V_elem(Delta_t,n1,n2);
        }
    }
    return V_mat;
}


///
///initialize A,B,S2,V in tree1
void evolution::init_tree1_mats()
{
//initialize tree1_A_mat_all
    tree1_A_mat_all.reserve(6);
    // for (const auto& ind : this->U1_inds)
    // {
    //     arma::cx_dmat A_tmp=
    // }

}