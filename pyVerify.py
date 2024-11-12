import numpy as np

N1=270
N2=600

j1H=0
j2H=2
g0=10
omegam=3
omegap=1
omegac=10
er=100
thetaCoef=0.1


L1=5
L2=8
r=np.log(er)
theta=thetaCoef*np.pi
Deltam=omegam-omegap

e2r=er**2

lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam

dx1=2.0*L1/(N1)
dx2=2.0*L2/(N2)

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])

x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])

D=lmd**2*np.sin(theta)**2+omegap**2

mu=lmd*np.cos(theta)+Deltam

def rho(x1):
    return omegac*x1**2-1/2

def s2(x1,x2,tau):
    tmp=x2*np.exp(lmd*np.sin(theta)*tau)-g0*lmd*np.sin(theta)/D*np.sqrt(2/omegam)*rho(x1)*np.sin(omegap*tau)*np.exp(lmd*np.sin(theta)*tau)\
    +g0*omegap/D*np.sqrt(2/omegam)*rho(x1)*np.cos(omegap*tau)*np.exp(lmd*np.sin(theta)*tau)\
    -g0*omegap/D*np.sqrt(2/omegam)*rho(x1)

    return tmp


n1=10
n2=134
tauTmp=0.04

print(s2(x1ValsAll[n1],x2ValsAll[n2],tauTmp))