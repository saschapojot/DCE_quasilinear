//
// Created by polya on 10/13/24.
//

#include "evolution.hpp"


double evolution::f1(int n1)
{
    double x1TmpSquared = x1ValsAllSquared[n1];
    double x1Tmp = x1ValsAll[n1];

    double valTmp = std::exp(-0.5 * omegac * x1TmpSquared)
        * std::hermite(this->j1H, std::sqrt(omegac) * x1Tmp);


    return valTmp;
}


double evolution::f2(int n2)
{
    double x2TmpSquared = x2ValsAllSquared[n2];
    double x2Tmp = x2ValsAll[n2];


    //    double valTmp=std::exp(-0.5 * omegam*std::exp(-2.0*r) * x2TmpSquared)
    //                  *std::hermite(this->jH2,std::sqrt(omegam*std::exp(-2.0*r))*x2Tmp);
    double valTmp = std::exp(-0.5 * omegam * x2TmpSquared)
        * std::hermite(this->j2H, std::sqrt(omegam) * x2Tmp);

    return valTmp;
}


void evolution::init_psi0()
{
    arma::cx_dcolvec vec1(N1);
    arma::cx_drowvec vec2(N2);
    for (int n1 = 0; n1 < N1; n1++)
    {
        vec1(n1) = f1(n1);
    }
    for (int n2 = 0; n2 < N2; n2++)
    {
        vec2(n2) = f2(n2);
    }
    this->psi0_arma = arma::kron(vec1, vec2);
    std::complex<double> nm(arma::norm(psi0_arma, "fro"), 0);
    psi0_arma /= nm;

    // std::cout<<"norm="<<arma::norm(psi0_arma,"fro")<<std::endl;
}

double evolution::rho(const double& x1)
{
    double val = omegac * std::pow(x1, 2.0) - 0.5;

    return val;
}

void evolution::init()
{
    this->init_psi0();
    this->construct_S_mat_spatial();

    this->init_mats_in_trees();
    std::cout<<"init finished"<<std::endl;
}


std::complex<double> evolution::A(const double& x1, const double& x2, const double& tau)
{
    double rhoVal = this->rho(x1);
    double rhoVal_squared = std::pow(rhoVal, 2.0);

    std::complex<double> part0 = 1i * this->P1(rhoVal) * tau;

    std::complex<double> part1 = 1i * F2 * rhoVal * x2 * std::cos(omegap * tau);

    std::complex<double> part2 = 1i * F3 * rhoVal * x2 * std::sin(omegap * tau);

    std::complex<double> part3 = 1i * F4 * rhoVal_squared * std::cos(2 * omegap * tau);

    std::complex<double> part4 = 1i * F5 * rhoVal_squared * std::sin(2.0 * omegap * tau);

    std::complex<double> part5 = 1i * F6 * rhoVal_squared;

    std::complex<double> part6 = 1i * F7 * std::pow(x2, 2.0);

    std::complex<double> part7(0.5 * lmd * std::sin(theta) * tau, 0);

    std::complex<double> A_Val = part0 + part1 + part2 + part3
        + part4 + part5 + part6 + part7;

    return A_Val;
}


double evolution::P1(const double& rhoVal)
{
    double val = 0.25 * omegac + 0.5 * Deltam - 0.5 * omegac * rhoVal + (2.0 * omegap - mu) / (2.0 * D) * std::pow(
        g0 * rhoVal, 2.0);

    return val;
}


std::complex<double> evolution::B(const double& x1, const double& x2, const double& tau)
{
    double rhoVal = this->rho(x1);
    double rhoVal_squared = std::pow(rhoVal, 2.0);

    double expVal = std::exp(lmd * std::sin(theta) * tau);

    double expVal_squared = std::exp(2.0 * lmd * std::sin(theta) * tau);

    std::complex<double> part0 = 1i * R1 * rhoVal_squared;

    std::complex<double> part1 = 1i * R2 * rhoVal * x2 * expVal;

    std::complex<double> part2 = 1i * R3 * rhoVal_squared * std::sin(omegap * tau) * expVal;

    std::complex<double> part3 = 1i * R4 * rhoVal_squared * std::cos(omegap * tau) * expVal;

    std::complex<double> part4 = 1i * (R5 * std::pow(x2, 2.0) + R6 * rhoVal_squared) * expVal_squared;


    std::complex<double> part5 = 1i * R7 * rhoVal * x2 * std::sin(omegap * tau) * expVal_squared;

    std::complex<double> part6 = 1i * R8 * rhoVal * x2 * std::cos(omegap * tau) * expVal_squared;

    std::complex<double> part7 = 1i * R9 * rhoVal_squared * std::cos(2.0 * omegap * tau) * expVal_squared;

    std::complex<double> part8 = 1i * R10 * rhoVal_squared * std::sin(2.0 * omegap * tau) * expVal_squared;

    std::complex<double> B_Val = part0 + part1 + part2 + part3
        + part4 + part5 + part6 + part7 + part8;

    return B_Val;
}


arma::cx_dmat evolution::construct_S_mat(const double& tau)
{
    // this->construct_S_mat_spatial();
    double exp_part = std::exp(lmd * std::sin(theta) * tau);

    double sin_val = std::sin(omegap * tau);

    double cos_val = std::cos(omegap * tau);

    arma::dmat S2_mat = S2_mat_part1 * exp_part + S2_mat_part2 * sin_val * exp_part
        + S2_mat_part3 * cos_val * exp_part + S2_mat_part4;


    return arma::conv_to<arma::cx_dmat>::from(S2_mat);
}


void evolution::construct_S_mat_spatial()
{
    this->S2_mat_part1 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part2 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part3 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part4 = arma::dmat(N1, N2, arma::fill::zeros);

    //begin initializing S2_mat_part1
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part1(n1, n2) = this->x2ValsAll[n2];
        } //end n2
    } //end n1, end initializing S2_mat_part1

    //begin initializing S2_mat_part2
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part2(n1, n2) = -g0 * lmd * std::sin(theta) / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part2

    //begin initializing S2_mat_part3
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part3(n1, n2) = g0 * omegap / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part3


    //begin initializing S2_mat_part4
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part4(n1, n2) = -g0 * omegap / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part4
}


arma::cx_dmat evolution::construct_A_mat(const double& tau)
{
    arma::cx_dmat A_mat(N1, N2, arma::fill::zeros);

    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            double x1n1 = this->x1ValsAll[n1];
            double x2n2 = this->x2ValsAll[n2];

            A_mat(n1, n2) = this->A(x1n1, x2n2, tau);
        } //end n2
    } //end n1
    return A_mat;
}

arma::cx_dmat evolution::construct_B_mat(const double& tau)
{
    arma::cx_dmat B_mat(N1, N2, arma::fill::zeros);
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            double x1n1 = this->x1ValsAll[n1];
            double x2n2 = this->x2ValsAll[n2];
            B_mat(n1, n2) = this->B(x1n1, x2n2, tau);
        } //end n2
    } //end n1

    return B_mat;
}


std::complex<double>* evolution::cx_dmat_2_complex_ptr(const arma::cx_dmat& psi)
{
    return const_cast<std::complex<double>*>(psi.memptr());
}


void evolution::Phi_2_c_arma()
{
    // Phi to d_ptr, column fft
    fftw_execute(plan_col_Phi_2_d_ptr);

    //d_ptr to d_arma
    this->d_arma = arma::cx_dmat(d_ptr, N1, N2, true);

    //multiply jth row of d_arma with multiplier_of_d_arma[j]
    for (int j = 0; j < N2; j++)
    {
        d_arma.col(j) *= multiplier_of_d_arma[j];
    }

    c_arma = d_arma * one_over_N2;
    // return c_arma;
}

void evolution::step_U1(arma::cx_dmat & Psi_arma,
        const std::vector<arma::cx_dmat>&tree_x_exp_A_minus_B,
        const std::vector<arma::cx_cube>& tree_x_exp_S2_cube,
        const int& j)
{

    //Psi to Phi
    std::memcpy(this->Phi, Psi_arma.memptr(), sizeof(std::complex<double>) * N2 * N1);

    this->Phi_2_c_arma();
    const auto & one_exp_A_minus_B=tree_x_exp_A_minus_B[j];
   const  auto & one_exp_S2_cube=tree_x_exp_S2_cube[j];
    // interpolation step
    for (int i=0;i<N1;i++)
    {
        this->psi_tilde.row(j)=c_arma.row(j)*one_exp_S2_cube.slice(j);
    }//end for
    // evolution step
    Psi_arma=psi_tilde%one_exp_A_minus_B;
}
///
/// @param Psi_arma wavefunction
/// @param tree_x_V_mat_all vector for V mat
/// /// @param  j: which one of tree_x_V_mat_all to use
void evolution::step_U2(arma::cx_dmat& Psi_arma, const std::vector<arma::cx_dmat>& tree_x_V_mat_all, const int& j)
{
    //Psi to Phi
    std::memcpy(this->Phi, Psi_arma.memptr(), sizeof(std::complex<double>) * N2 * N1);


    //Phi to D_widehat
    fftw_execute(plan_2d_fft_Phi_2_D_widehat);

    //D_widehat to F_widehat_arma
    this->F_widehat_arma = arma::cx_dmat(this->D_widehat, N1, N2, true);

    //F_widehat_arma to G_widehat_arma

    V_tmp = tree_x_V_mat_all[j];

    this->G_widehat_arma = F_widehat_arma % V_tmp;

    std::memcpy(this->I_widehat, G_widehat_arma.memptr(), sizeof(std::complex<double>) * N2 * N1);


    fftw_execute(plan_2d_ifft_I_widehat_2_J);

    Psi_arma = arma::cx_dmat(J, N1, N2, true) * normalizing_factor2d;
}
void evolution::tree1_evolution(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,0);
    this->step_U2(Psi_arma,tree1_V_mat_all,0);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,2);
    this->step_U2(Psi_arma,tree1_V_mat_all,1);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,4);
    this->step_U2(Psi_arma,tree1_V_mat_all,2);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,5);
}

void evolution::tree2_evolution(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,0);
    this->step_U2(Psi_arma,tree2_V_mat_all,0);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,2);
    this->step_U2(Psi_arma,tree2_V_mat_all,1);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,4);
    this->step_U2(Psi_arma,tree2_V_mat_all,2);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,5);
}

void evolution::tree3_evolution(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,0);
    this->step_U2(Psi_arma,tree3_V_mat_all,0);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,2);
    this->step_U2(Psi_arma,tree3_V_mat_all,1);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,4);
    this->step_U2(Psi_arma,tree3_V_mat_all,2);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,5);
}

void evolution::evolution_1_step(arma::cx_dmat & Psi_arma)
{
    this->tree1_evolution(Psi_arma);
    this->tree2_evolution(Psi_arma);
    this->tree3_evolution(Psi_arma);
}
///
/// @param Delta_t time step
/// @param n1
/// @param n2
/// @return element of V
std::complex<double> evolution::V_elem(const double& Delta_t, const int& n1, const int& n2)
{
    double tmp = (lmd * std::cos(theta) - Deltam) / (2.0 * omegam) * k2ValsAllSquared_fft[n2] - 0.5 *
        k1ValsAllSquared_fft[n1];
    tmp *= Delta_t;

    std::complex<double> to_exp = std::complex<double>(0.0, tmp);

    std::complex<double> elem_tmp = std::exp(to_exp);

    return elem_tmp;
}

arma::cx_dmat evolution::construct_V_mat(const double& Delta_t)
{
    arma::cx_dmat V_mat(N1, N2, arma::fill::zeros);
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            V_mat(n1, n2) = this->V_elem(Delta_t, n1, n2);
        }
    }
    return V_mat;
}

///
/// @param vec
/// @return the product of elements in vec
double evolution::vec_prod(const std::vector<double>& vec)
{
    double rst = 1;
    for (const auto& elem : vec)
    {
        rst *= elem;
    }
    return rst;
}

///
///initialize A,B,S2,V in tree1
void evolution::init_tree1_mats()
{
    // std::cout<<"this->U1_inds.size()="<<this->U1_inds.size()<<std::endl;
    // U1 part
    //initialize tree1_A_mat_all
    // std::cout<<"entering init_tree1_mats()"<<std::endl;
    tree1_A_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree1[ind]);
        double tauTmp_coef = this->vec_prod(tree1[ind]);
        // std::cout<<"tauTmp_coef="<<tauTmp_coef<<std::endl;
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat A_tmp = construct_A_mat(tau_tmp);
        tree1_A_mat_all.push_back(A_tmp);
    } //end initializing tree1_A_mat_all


    //initialize tree1_B_mat_all
    tree1_B_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree1[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat B_tmp = construct_B_mat(tau_tmp);
        tree1_B_mat_all.push_back(B_tmp);
    } //end initializing tree1_B_mat_all
    //initializing std::vector<arma::cx_dmat>tree1_exp_A_minus_B;
    tree1_exp_A_minus_B.reserve(this->U1_inds.size());
    for (int i = 0; i < tree1_B_mat_all.size(); i++)
    {
        arma::cx_dmat A_tmp = tree1_A_mat_all[i];
        arma::cx_dmat B_tmp = tree1_B_mat_all[i];
        arma::cx_dmat exp_tmp = arma::exp(A_tmp - B_tmp);
        tree1_exp_A_minus_B.push_back(exp_tmp);
    }
    //end initializing std::vector<arma::cx_dmat>tree1_exp_A_minus_B;
    //initialize tree1_S2_mat_all
    tree1_S2_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree1[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat S2_tmp = construct_S_mat(tau_tmp);
        tree1_S2_mat_all.push_back(S2_tmp);
    } //end initializing  tree1_S2_mat_all

    //tree1, exp S2 cubes
    tree1_exp_S2_cube.reserve(this->U1_inds.size());
    for (const auto& S2 : tree1_S2_mat_all)
    {
        tree1_exp_S2_cube.push_back(construct_1_cube(S2));
    }
    //end tree1, exp S2 cubes
    // end U1 part
    // std::cout<<"entering tree1, U2"<<std::endl;

    // U2 part
    //initialize tree1_V_mat_all
    tree1_V_mat_all.reserve(this->U2_inds.size());
    for (const auto& ind : this->U2_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree1[ind]);
        double Delta_t_coef_tmp = this->vec_prod(tree1[ind]);
        double Delta_tTmp = Delta_t_coef_tmp * this->dt;
        arma::cx_dmat V_tmp = construct_V_mat(Delta_tTmp);
        tree1_V_mat_all.push_back(V_tmp);
    }
    // std::cout<<"finished tree1"<<std::endl;
    //end initializing  tree1_V_mat_all
    // end U2 part
}

///
///initialize A,B,S2,V in tree2
void evolution::init_tree2_mats()
{
    // U1 part
    //initialize tree2_A_mat_all
    tree2_A_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree2[ind]);
        double tauTmp_coef = this->vec_prod(tree2[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat A_tmp = construct_A_mat(tau_tmp);
        tree2_A_mat_all.push_back(A_tmp);
    } //end initializing tree2_A_mat_all

    //initialize tree2_B_mat_all
    tree2_B_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree2[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat B_tmp = construct_B_mat(tau_tmp);
        tree2_B_mat_all.push_back(B_tmp);
    } //end initializing tree2_B_mat_all
    tree2_exp_A_minus_B.reserve(this->U1_inds.size());
    // initializing tree2_exp_A_minus_B
    for (int i = 0; i < tree2_B_mat_all.size(); i++)
    {
        arma::cx_dmat A_tmp = tree2_A_mat_all[i];
        arma::cx_dmat B_tmp = tree2_B_mat_all[i];
        arma::cx_dmat exp_tmp = arma::exp(A_tmp - B_tmp);
        tree2_exp_A_minus_B.push_back(exp_tmp);
    }

    //end initializing tree2_exp_A_minus_B

    //initialize tree2_S2_mat_all
    tree2_S2_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree2[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat S2_tmp = construct_S_mat(tau_tmp);
        tree2_S2_mat_all.push_back(S2_tmp);
    } //end initializing  tree2_S2_mat_all

    //tree2, exp S2 cubes
    tree2_exp_S2_cube.reserve(this->U1_inds.size());
    for (const auto& S2 : tree2_S2_mat_all)
    {
        tree2_exp_S2_cube.push_back(construct_1_cube(S2));
    }
    //end tree2, exp S2 cubes
    // end U1 part


    // U2 part
    //initialize tree2_V_mat_all
    // std::cout<<"this->U2_inds.size()="<<this->U2_inds.size()<<std::endl;
    tree2_V_mat_all.reserve(this->U2_inds.size());
    for (const auto& ind : this->U2_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree2[ind]);
        double Delta_t_coef_tmp = this->vec_prod(tree2[ind]);
        // std::cout<<"Delta_t_coef_tmp="<<Delta_t_coef_tmp<<std::endl;
        double Delta_tTmp = Delta_t_coef_tmp * this->dt;
        arma::cx_dmat V_tmp = construct_V_mat(Delta_tTmp);
        tree2_V_mat_all.push_back(V_tmp);
    } //end initializing  tree2_V_mat_all
    // end U2 part
}

///
///initialize A,B,S2,V in tree3
void evolution::init_tree3_mats()
{
    // U1 part
    //initialize tree3_A_mat_all
    tree3_A_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree3[ind]);
        double tauTmp_coef = this->vec_prod(tree3[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat A_tmp = construct_A_mat(tau_tmp);
        tree3_A_mat_all.push_back(A_tmp);
    } //end initializing tree3_A_mat_all

    //initialize tree3_B_mat_all
    tree3_B_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree3[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat B_tmp = construct_B_mat(tau_tmp);
        tree3_B_mat_all.push_back(B_tmp);
    } //end initializing tree3_B_mat_all
    // initializing tree3_exp_A_minus_B
    tree3_exp_A_minus_B.reserve(this->U1_inds.size());
    for (int i = 0; i < tree3_B_mat_all.size(); i++)
    {
        arma::cx_dmat A_tmp = tree3_A_mat_all[i];
        arma::cx_dmat B_tmp = tree3_B_mat_all[i];
        arma::cx_dmat exp_tmp = arma::exp(A_tmp - B_tmp);
        tree3_exp_A_minus_B.push_back(exp_tmp);
    }
    //end initializing tree3_exp_A_minus_B
    //initialize tree3_S2_mat_all
    tree3_S2_mat_all.reserve(this->U1_inds.size());
    for (const auto& ind : this->U1_inds)
    {
        double tauTmp_coef = this->vec_prod(tree3[ind]);
        double tau_tmp = tauTmp_coef * this->dt;
        arma::cx_dmat S2_tmp = construct_S_mat(tau_tmp);
        tree3_S2_mat_all.push_back(S2_tmp);
    } //end initializing tree3_S2_mat_all

    //tree3, exp S2 cubes
    tree3_exp_S2_cube.reserve(U1_inds.size());
    for (const auto& S2 : tree3_S2_mat_all)
    {
        tree3_exp_S2_cube.push_back(construct_1_cube(S2));
    }
    //end tree3, exp S2 cubes

    // end U1 part
    // std::cout<<"entering tree3, U2"<<std::endl;

    // U2 part
    //initialize tree3_V_mat_all
    // tree3_V_mat_all.reserve(this->U2_inds.size());
    for (const auto& ind : this->U2_inds)
    {
        // std::cout<<"ind="<<ind<<", ";
        // printVec(tree3[ind]);
        double Delta_t_coef_tmp = this->vec_prod(tree3[ind]);
        // std::cout<<"Delta_t_coef_tmp="<<Delta_t_coef_tmp<<std::endl;
        double Delta_tTmp = Delta_t_coef_tmp * this->dt;
        arma::cx_dmat V_tmp = construct_V_mat(Delta_tTmp);
        tree3_V_mat_all.push_back(V_tmp);
    } //end initializing  tree3_V_mat_all
}

void evolution::init_mats_in_trees()
{
    this->init_tree1_mats();
    this->init_tree2_mats();
    this->init_tree3_mats();
}


arma::cx_dmat evolution::make_1_slice_in_cube(const int& j, const arma::cx_dmat& S2_mat)
{
    arma::cx_rowvec S2_jth_row = S2_mat.row(j);
    arma::cx_dmat k2_S2_j = k2Vals_arma_col_vec * S2_jth_row;
    // k2_S2_j.print("k2_S2_j:");

    return arma::exp(1i * k2_S2_j);
}


arma::cx_cube evolution::construct_1_cube(const arma::cx_dmat& S2_mat)
{
    arma::cx_cube one_cube(N2, N2, N1);
    for (int j = 0; j < N1; j++)
    {
        arma::cx_dmat one_slice = this->make_1_slice_in_cube(j, S2_mat);
        one_cube.slice(j) = one_slice;
    }
    // std::cout<<"finished constructing cube"<<std::endl;
    return one_cube;
}


void evolution::tree1_evolution_H1R_only(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,0);
    //this->step_U2(Psi_arma,tree1_V_mat_all,0);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,2);
    //this->step_U2(Psi_arma,tree1_V_mat_all,1);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,4);
    //this->step_U2(Psi_arma,tree1_V_mat_all,2);
    this->step_U1(Psi_arma,tree1_exp_A_minus_B,tree1_exp_S2_cube,5);
}


void evolution::tree2_evolution_H1R_only(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,0);
   // this->step_U2(Psi_arma,tree2_V_mat_all,0);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,2);
    //this->step_U2(Psi_arma,tree2_V_mat_all,1);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,4);
    //this->step_U2(Psi_arma,tree2_V_mat_all,2);
    this->step_U1(Psi_arma,tree2_exp_A_minus_B,tree2_exp_S2_cube,5);
}


void evolution::tree3_evolution_H1R_only(arma::cx_dmat & Psi_arma)
{
    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,0);
    //this->step_U2(Psi_arma,tree3_V_mat_all,0);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,1);
    //////////////////////////////////////////////////////

    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,2);
    //this->step_U2(Psi_arma,tree3_V_mat_all,1);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,3);

    //////////////////////////////////////////////////////
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,4);
    //this->step_U2(Psi_arma,tree3_V_mat_all,2);
    this->step_U1(Psi_arma,tree3_exp_A_minus_B,tree3_exp_S2_cube,5);

}

void evolution::evolution_1_step_H1R_only(arma::cx_dmat & Psi_arma)
{
    this->tree1_evolution_H1R_only(Psi_arma);
    this->tree2_evolution_H1R_only(Psi_arma);
    this->tree3_evolution_H1R_only(Psi_arma);
}

void evolution::save_complex_array_to_pickle(std::complex<double> ptr[],
                                    int size,
                                    const std::string& filename)
{
    // Initialize Python interpreter if it is not already initialized.
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize();  // Initialize NumPy
    }

    try
    {
        // Import the pickle module and retrieve the dumps function.
        bp::object pickle = bp::import("pickle");
        bp::object pickle_dumps = pickle.attr("dumps");

        // Convert the C++ complex array to a NumPy array.
        np::ndarray numpy_array = np::from_data(
            ptr,                                           // Raw pointer to data
            np::dtype::get_builtin<std::complex<double>>(),// NumPy dtype for std::complex<double>
            bp::make_tuple(size),                          // Shape: 1D array with "size" elements
            bp::make_tuple(sizeof(std::complex<double>)),  // Stride: size of one element
            bp::object()                                   // No base object provided
        );

        // Serialize the NumPy array using pickle.dumps.
        bp::object serialized_obj = pickle_dumps(numpy_array);
        std::string serialized_str = bp::extract<std::string>(serialized_obj);

        // Write the serialized data to a file.
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Optional debug output.
        // std::cout << "Complex array successfully serialized and saved to " << filename << std::endl;
    }
    catch (const bp::error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred while saving complex array." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}


void evolution::run_and_save_H1R_only()
{
    this->Psi=this->psi0_arma;
    std::string outPath="./outData/group"+std::to_string(groupNum)+"/row"
    +std::to_string(rowNum)+"/wavefunction/";
    std::string outFileName;
    if (!fs::is_directory(outPath) || !fs::exists(outPath))
    {
        fs::create_directories(outPath);
    }//end creating outPath
    std::cout<<"created out dir"<<std::endl;
    for (int j=0;j<10;j++)
    {
        this->evolution_1_step_H1R_only(Psi);
        if (j%1==0)
        {   std::cout<<"saving at step "<<j<<std::endl;
            outFileName=outPath+"/at_time_step_"+std::to_string(j+1);


            this->save_complex_array_to_pickle(Psi.memptr(),N1*N2,outFileName);
        }//end save to file
    }//end for
}