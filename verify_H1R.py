import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys

#this script verifies H1R part of the numerical subroutines

#this script applies to sin(theta)!=0 case

if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParams/inParams"+str(groupNum)+".csv"
# print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]


j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification

thetaCoef=float(oneRow.loc["thetaCoef"])
theta=thetaCoef*np.pi

print("j1H="+str(j1H)+", j2H="+str(j2H)+", g0="+str(g0)\
      +", omegam="+str(omegam)+", omegap="+str(omegap)\
      +", omegac="+str(omegac)+", er="+str(er)+", thetaCoef="+str(thetaCoef))
r=np.log(er)
Deltam=omegam-omegap
print(f"Deltam={Deltam}")
print(f"r={r}")
lmd=Deltam*np.tanh(2*r)
print(f"lmd={lmd}")
D=lmd**2*(np.sin(theta))**2+omegap**2
print(f"D={D}")


mu=lmd*np.cos(theta)+Deltam
print(f"mu={mu}")
def rho(x1):
    return omegac*x1**2-1/2


def P1(rho):
    val=1/4*omegac+1/2*Deltam-1/2*omegac*rho+(2*omegap-mu)/(2*D)*g0**2*rho**2

    return val

F2=g0*np.sqrt(2*omegam)*(2*lmd**2*(np.sin(theta))**2+omegap*mu)/(2*D*lmd*np.sin(theta))

F3=g0*np.sqrt(2*omegam)/D*(1/2*mu-omegap)

F4=g0**2*(2*D*lmd**2*(np.sin(theta))**2+lmd**2*mu*omegap*(np.sin(theta))**2+mu*omegap**3)/(4*D**2*lmd*omegap*np.sin(theta))


F5=g0**2*(2*omegap*D+mu*lmd**2*(np.sin(theta))**2-4*omegap*lmd**2*(np.sin(theta))**2+mu*omegap**2-4*omegap**3)/(4*omegap*D**2)


F6=g0**2*(8*lmd**2*omegap*(np.sin(theta))**2-4*lmd**2*mu*(np.sin(theta))**2+D*mu)/(4*lmd*np.sin(theta)*D**2)

F7=omegam*mu/(4*lmd*np.sin(theta))
