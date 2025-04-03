from sympy import *
import numpy as np
import sys
import pandas as pd
#this script verifies the correctness of the analytical solution of
# i\partial_{t}\psi=H_{1}^{r}\PSI

omegam,omegap=symbols(r"omega_m,omega_p",cls=Symbol,real=True)
omegac=symbols("omega_c",cls=Symbol,real=True)


r=symbols("r",cls=Symbol,real=True)

er=exp(r)

# theta=0.1*pi
theta=symbols("theta",cls=Symbol,real=True)
g0=symbols("g0",cls=Symbol,real=True)

Deltam=omegam-omegap

lmd=Deltam*tanh(2*r)

half=Rational(1,2)

x1,x2=symbols("x1,x2",cls=Symbol,real=True)

rho=omegac*x1**2-half

D=lmd**2*(sin(theta))**2+omegap**2
tau=symbols("tau",cls=Symbol,real=True)
mu=lmd*cos(theta)+Deltam

quarter=Rational(1,4)

P1=quarter*omegac+half*Deltam-half*omegac*rho\
    +(2*omegap-mu)/(2*D)*g0**2*rho**2


F2=g0*sqrt(2*omegam)*(2*lmd**2*(sin(theta))**2+omegap*mu)/(2*D*lmd*sin(theta))

F3=g0*sqrt(2*omegam)/D*(half*mu-omegap)

F4=g0**2*(2*D*lmd**2*(sin(theta))**2+lmd**2*mu*omegap*(sin(theta))**2+mu*omegap**3)\
   /(4*D**2*lmd*omegap*(sin(theta)))


F5=g0**2*(2*omegap*D+mu*lmd**2*(sin(theta))**2-4*omegap*lmd**2*(sin(theta))**2+mu*omegap**2-4*omegap**3)\
    /(4*omegap*D**2)


F6=g0**2*(8*lmd**2*omegap*(sin(theta))**2-4*lmd**2*mu*(sin(theta))**2+D*mu)\
   /(4*lmd*sin(theta)*D**2)


F7=omegam*mu/(4*lmd*sin(theta))


A=I*P1*tau+I*F2*rho*x2*cos(omegap*tau)+I*F3*rho*x2*sin(omegap*tau)\
    +I*F4*rho**2*cos(2*omegap*tau)+I*F5*rho**2*sin(2*omegap*tau)\
    +I*F6*rho**2\
    +I*F7*x2**2+half*lmd*sin(theta)*tau


R1=g0**2*lmd*sin(theta)/(2*omegap)*(D-omegap*mu)/D**2

R2=g0*lmd*sin(theta)/D*sqrt(2*omegam)


R3=-2*g0**2*lmd**2*(sin(theta))**2/D**2

R4=2*g0**2*omegap*lmd*sin(theta)/D**2

R5=omegam/(4*lmd*sin(theta))*mu

R6=mu*g0**2/(4*lmd*sin(theta)*D)

R7=-mu*g0/(2*D)*sqrt(2*omegam)

R8=mu*g0*omegap/(2*lmd*sin(theta)*D)*sqrt(2*omegam)

R9=mu*g0**2/(4*lmd*sin(theta)*D**2)*(omegap**2-lmd**2*(sin(theta))**2)


R10=-mu*omegap*g0**2/(2*D**2)

exp_val=exp(lmd*sin(theta)*tau)
exp_val2=exp_val**2

B=I*R1*rho**2\
    +I*R2*rho*x2*exp_val+I*R3*rho**2*sin(omegap*tau)*exp_val\
    +I*R4*rho**2*cos(omegap*tau)*exp_val\
    +I*(R5*x2**2+R6*rho**2)*exp_val2\
    +I*R7*rho*x2*sin(omegap*tau)*exp_val2\
    +I*R8*rho*x2*cos(omegap*tau)*exp_val2\
    +I*R9*rho**2*cos(2*omegap*tau)*exp_val2\
    +I*R10*rho**2*sin(2*omegap*tau)*exp_val2

s2=x2*exp_val-g0*lmd*sin(theta)/D*sqrt(2/omegam)*rho*sin(omegap*tau)*exp_val\
    +g0*omegap/D*sqrt(2/omegam)*rho*cos(omegap*tau)*exp_val\
    -g0*omegap/D*sqrt(2/omegam)*rho


def psi_init(x1,x2):
    return exp(-half*omegac*x1**2)*exp(-half*omegam*x2**2)


psi=psi_init(x1,s2)*exp(A-B)

c0=I*half*omegac+I*half*Deltam+I*half*g0*sqrt(2*omegam)*cos(omegap*tau)*x2-I*half*omegac**2*x1**2\
    -I*(half*lmd*omegam*cos(theta)+half*Deltam*omegam)*x2**2\
    -I*g0*omegac*sqrt(2*omegam)*cos(omegap*tau)*x1**2*x2\
    +half*lmd*sin(theta)

LHS=diff(psi,tau)\
    +(g0*omegac*sqrt(2/omegam)*sin(omegap*tau)*x1**2-half*g0*sqrt(2/omegam)*sin(omegap*tau)-lmd*sin(theta)*x2)*diff(psi,x2)




RHS=c0*psi


if len(sys.argv)!=3:
    print("wrong number of arguments")


groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParams/inParams"+str(groupNum)+".csv"

dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]
j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0_val=float(oneRow.loc["g0"])

omegam_val=float(oneRow.loc["omegam"])
omegap_val=float(oneRow.loc["omegap"])
omegac_val=float(oneRow.loc["omegac"])
er_val=float(oneRow.loc["er"])#magnification

r_val=np.log(er_val)
thetaCoef_val=float(oneRow.loc["thetaCoef"])
theta_val=thetaCoef_val*np.pi

Deltam_val=omegam_val-omegap_val
lmd_val=Deltam_val*np.tanh(2*r_val)

D_val=lmd_val**2*(np.sin(theta_val))**2+omegap_val**2

mu_val=lmd_val*np.cos(theta_val)+Deltam_val


x1_val = 0.1
x2_val = 0.2
tau_val = 0.1
# Create a substitution dictionary including all values
param_subs = {
    g0: g0_val,
    omegam: omegam_val,
    omegap: omegap_val,
    omegac: omegac_val,
    r: r_val,
    theta: theta_val,
    x1: x1_val,
    x2: x2_val,
    tau: tau_val
}


lambda_vars = (tau, x1, x2, g0, omegac, omegam, omegap, r, theta)

# Generate the lambdified functions using numpy as the module:
LHS_func = lambdify(lambda_vars, LHS, modules="numpy")
RHS_func = lambdify(lambda_vars, RHS, modules="numpy")
lhs_numeric = LHS_func(tau_val, x1_val, x2_val, g0_val, omegac_val, omegam_val, omegap_val, r_val, theta_val)
rhs_numeric = RHS_func(tau_val, x1_val, x2_val, g0_val, omegac_val, omegam_val, omegap_val, r_val, theta_val)

print("Numerical evaluation of LHS:")
print(lhs_numeric)
print("\nNumerical evaluation of RHS:")
print(rhs_numeric)