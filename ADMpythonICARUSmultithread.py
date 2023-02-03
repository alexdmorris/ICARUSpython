# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:13:45 2022

@author: admorris
"""

import math
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
import datetime
#import multiprocessing
#import threading
#from scipy.integrate import nquad
#from mpmath import mp
import os
from scipy.stats import qmc
from decimal import Decimal

"""
Defining constants
"""
starttime = time.time();
now = datetime.datetime.now()
print("Start time is", now.time())
re = 2.8179403262e-15; #electron radius [m]
clight = 299792458.0; #speed of light [m/s]
hbar = 6.5821e-16; # reduced planck constant [eV.s]
me = 0.51099895e6; #mass of electron in eV
elecharge = 1.60217662e-19; # electron charge [C]
#mp.dps = 50
phMeVnC = 1e6 * 1e-9 / elecharge;
phkeVnC = 1e3 * 1e-9 / elecharge;
runmontecarlo = 1;
plotandsave = 1;
runmontecarlotest = 0;
runqmc = 0;
NMC = int(100e6);
QNMC = 27 # total number of quasi montecarlo points is 2^QNMC 27 for 134217728
Elowset = 1.47e6 #manually set the lower limit for the energy range set to 0 for automatic lower limit
"""
Defining equation functions
"""

"""Polarisation"""
tau = 0
@jit
def phif(thetax,thetay):
    if thetax != 0: return math.acos(thetax/(thetax**2 +thetay**2)**0.5)
    else: return math.acos(0)
Pt = 1
Pc = 0

"""Case for variables"""
Case = "FEBE250CASEBtesting"
if Case == "FEBE250CASEB":
    Ee_ = 250e6
    Q_ = 100e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.070e-3
    crossingangle_ = 0
    enx_ = 1e-6
    Betax_ = 1.23
    eny_ = 1e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 8e-5
    dk_ = 0.03185
    L_ = 10
    npts_ = 10
    Energyrange = "MeV"
    print('Case is FEBE250CASEB')
elif Case == "CBETA150FWHM":
    Ee_ = 150e6
    Q_ = 32e-12
    Epulse_ = 62e-6
    wavelength_ = 1064e-9
    collimationangle_ = 0.256e-3
    crossingangle_ = 0
    enx_ = 0.3e-6
    Betax_ = 0.269211
    eny_ = 0.3e-6
    Betay_ = 0.269211
    sigmaL_ = 25e-6
    dEe_ = 5e-4
    dk_ = 6.57e-4
    L_ = 10
    npts_ = 71
    Energyrange = "keV"
    print("Case is CBETA150FWHM")
elif Case == "FEBE600CASEB":
    
    Ee_ = 600e6
    Q_ = 100e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.024e-3
    crossingangle_ = 0
    enx_ = 1e-6
    Betax_ = 1.23
    eny_ = 1e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 8e-5
    dk_ = 0.03185
    L_ = 10
    npts_ = 71
    Energyrange = "MeV"
    print('Case is ', Case)
    
elif Case == "FEBE250CASEBtesting":
    Ee_ = 250e6
    Q_ = 100e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.070e-3
    crossingangle_ = 0
    enx_ = 1e-6
    Betax_ = 1.23
    eny_ = 1e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 8e-5
    dk_ = 0.03185
    L_ = 10
    npts_ = 10
    Energyrange = "MeV"
    print('Case is FEBE250CASEBtesting')
    
else:
    Ee_ = 250e6
    Q_ = 10e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.070e-3
    crossingangle_ = 0
    enx_ = 1e-6
    Betax_ = 1.23
    eny_ = 1e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 8e-5
    dk_ = 0.03185
    L_ = 10
    npts_ = 400
    
    print('Default values for FEBE parameters used')

#Simulation setup
#NKERNEL = 20
#NQMC = 5
rmsBW = 0.3/100
Egammatest = 1.47e6


#Lorentz gamma factor
@jit
def gamma(Ee):
    return (Ee + me)/me

#Relative velocity
@jit
def Beta(Ee):
    return (1 - gamma(Ee)**-2)**0.5

#Photon energy
@jit
def EL(wavelength):
    return 2*np.pi*hbar*clight/wavelength

#Scattered photon energy
@jit
def Egammaset(wavelength,Ee,crossingangle,obsangle):
    return EL(wavelength)*(1 - Beta(Ee)*np.cos(np.pi-crossingangle))/(1 - Beta(Ee)*np.cos(obsangle) + EL(wavelength)*(1 - np.cos(np.pi - crossingangle - obsangle))/Ee)

"""Sun preliminaries"""
@jit
def Ne(Q):
    return Q/elecharge
@jit
def NL(Epulse,wavelength):
    return Epulse/(EL(wavelength)*elecharge)
@jit
def ex(Ee,enx):
    return enx/(Beta(Ee)* gamma(Ee))
@jit
def ey(Ee,eny):
    return eny/(Beta(Ee)* gamma(Ee))
@jit
def zr(wavelength,sigmaL):
    return 4*np.pi*(sigmaL**2)/wavelength
@jit
def sigmae(dEe,Ee):
    return dEe*Ee
@jit
def klaser(wavelength):
    return 2*np.pi/wavelength
@jit
def sigmak(dk,wavelength):
    return dk*klaser(wavelength)
@jit
def sigmaelectron(BetaIP,Ee,en):
    return (BetaIP*ex(Ee,en))**0.5

"""Collimation"""
@jit
def R(L,collimationangle):
    return L*collimationangle
@jit
def xtheta(L,collimationangle):
    return L*collimationangle
@jit
def ytheta(L,collimationangle,xd):
    return ((R(L,collimationangle)**2 - xd**2)**0.5).real


"""Sun intermediatiries"""

@jit  
def gammabar(wavelength,Egamma,thetax,thetay):
    return ((2*Egamma*EL(wavelength))*((1+(me**2*(4*EL(wavelength) - Egamma*(thetax**2 + thetay**2))) \
            /(4*Egamma*EL(wavelength)**2))**0.5 + 1)) \
            /(me*(4*EL(wavelength) - Egamma*(thetax**2 + thetay**2)))
    
    
@jit           
def Xix(Betax,L,k,Ee,enx,wavelength,sigmaL):
    return 1 + (Betax/L)**2 + (2*k*Betax*ex(Ee,enx))/zr(wavelength,sigmaL)

@jit
def Xiy(Betay,L,k,Ee,eny,wavelength,sigmaL):
    return 1 + (Betay/L)**2 + (2*k*Betay*ey(Ee,eny))/zr(wavelength,sigmaL)

@jit
def Zetax(k, Betax, Ee, enx, wavelength, sigmaL):
    return 1 + (2*k*Betax*ex(Ee, enx))/zr(wavelength, sigmaL)

@jit
def Zetay(k, Betay, Ee, eny, wavelength, sigmaL):
    return 1 + (2*k*Betay*ey(Ee, eny))/zr(wavelength, sigmaL)
@jit           
def sigmathetax(Ee, enx, Betax, L, k, wavelength, sigmaL):
    return ((ex(Ee, enx)*Xix(Betax, L, k, Ee, enx, wavelength, sigmaL))/(Betax*Zetax(k, Betax, Ee, enx, wavelength, sigmaL)))**0.5
@jit
def sigmathetay(Ee, eny, Betay, L, k, wavelength, sigmaL):
    return ((ey(Ee, eny)*Xiy(Betay, L, k, Ee, eny, wavelength, sigmaL))/(Betay*Zetay(k, Betay, Ee, eny, wavelength, sigmaL)))**0.5
@jit
def sigmagamma(dEe,Ee):
    return sigmae(dEe,Ee)/me
@jit
def thetaymax(wavelength,Egamma):
    return (4*EL(wavelength)/Egamma)**0.5
@jit
def thetaxmax(wavelength,Egamma, thetay):
    return (4*EL(wavelength)/Egamma - thetay**2)**0.5
@jit
def kmin(wavelength,dk):
    return klaser(wavelength)*(1-3*dk)
@jit
def kmax(wavelength,dk):
    return klaser(wavelength)*(1+3*dk)
@jit
def prefactor(Q, Epulse, wavelength, sigmaL, dEe, Ee, dk, L):
    return (re**2*Ne(Q)*NL(Epulse, wavelength))/(4*np.pi**3*hbar*clight*zr(wavelength, sigmaL)*sigmagamma(dEe,Ee)*sigmak(dk, wavelength)*L**2)

""" Separated Intergral terms"""

@jit
def term1(k,Betax,Ee,enx,wavelength,Betay,eny,L,sigmaL):
    return (((Zetax(k,Betax,Ee,enx,wavelength,sigmaL)*Zetay(k,Betay,Ee,eny,wavelength,sigmaL))**0.5) \
            *sigmathetax(Ee,enx,Betax,L,k,wavelength,sigmaL)*sigmathetay(Ee,eny,Betay,L,k,wavelength,sigmaL))**-1
@jit
def term2(wavelength,Egamma,thetax,thetay):
    return gammabar(wavelength,Egamma,thetax,thetay)/(1 + (2*gammabar(wavelength,Egamma,thetax,thetay)*EL(wavelength))/me)
@jit
def term3(wavelength,Egamma,thetax,thetay,tau):
    return 0.25*((4*(gammabar(wavelength,Egamma,thetax,thetay)**2)*EL(wavelength)/(Egamma*(1 + (gammabar(wavelength,Egamma,thetax,thetay)**2)*(thetax**2 + thetay**2)))) \
               +  Egamma*(1 + (gammabar(wavelength,Egamma,thetax,thetay)**2)*(thetax**2 + thetay**2))/(4*gammabar(wavelength,Egamma,thetax,thetay)**2*EL(wavelength))) \
        - (1+Pt)*(np.cos(tau - phif(thetax,thetay))**2)*gammabar(wavelength,Egamma,thetax,thetay)**2*(thetax**2 + thetay**2) \
            /(1+gammabar(wavelength,Egamma,thetax,thetay)**2*(thetax**2 + thetay**2))**2
@jit
def term4(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL):
    return np.exp((-(((thetax - xd/L)**2)/(2*sigmathetax(Ee, enx, Betax, L, k, wavelength, sigmaL)**2)) \
                    - (((thetay - yd/L)**2)/(2*sigmathetay(Ee, eny, Betay, L, k, wavelength, sigmaL)**2)) \
                        - (((gammabar(wavelength, Egamma, thetax, thetay) - gamma(Ee))**2)/(2*sigmagamma(dEe, Ee)**2)) \
                            - (((k - klaser(wavelength))**2)/(2*sigmak(dk, wavelength)**2))))
        
@jit
def term4test(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL):
    return ((-(((thetax - xd/L)**2)/(2*sigmathetax(Ee, enx, Betax, L, k, wavelength, sigmaL)**2)) \
                    - (((thetay - yd/L)**2)/(2*sigmathetay(Ee, eny, Betay, L, k, wavelength, sigmaL)**2)) \
                        - (((gammabar(wavelength, Egamma, thetax, thetay) - gamma(Ee))**2)/(2*sigmagamma(dEe, Ee)**2)) \
                            - (((k - klaser(wavelength))**2)/(2*sigmak(dk, wavelength)**2))))

@jit
def intterm(k, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax, thetay, tau, xd, yd, dEe, dk):
    return term2(wavelength,Egamma,thetax,thetay)
        
"""
@jit
def intterm(k, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax, thetay, tau, xd, yd, dEe, dk):
    return term1(k,Betax,Ee,enx,wavelength,Betay,eny,L,sigmaL)*term2(wavelength,Egamma,thetax,thetay)\
        *term3(wavelength,Egamma,thetax,thetay,tau)*term4(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL)
"""
"""
@jit
def intterm(k, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax, thetay, tau, xd, yd, dEe, dk):
    return term4test(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL)
"""
@jit     
def Elow(rmsBW, wavelength, Ee, crossingangle):
    return (1 - rmsBW*6*(2*np.log(2))**0.5)*Egammaset(wavelength, Ee, crossingangle, 0)
@jit
def Ehigh(wavelength, Ee, dEe, crossingangle):
    return Egammaset(wavelength, Ee*(1+5*dEe), crossingangle, 0)
@jit
def Estep(wavelength, Ee, dEe, crossingangle, npts):
    return (Ehigh(wavelength, Ee, dEe, crossingangle) - Elow(rmsBW, wavelength, Ee, crossingangle))/npts


"""Creating an array of energies to loop through """

if Elowset != 0:
    Egammarange = np.linspace(start=Elowset, stop=Ehigh(wavelength_, Ee_, dEe_, crossingangle_), num=npts_)
else:
    Egammarange = np.linspace(start=Elow(rmsBW, wavelength_, Ee_, crossingangle_), stop=Ehigh(wavelength_, Ee_, dEe_, crossingangle_), num=npts_)
    
"""testing differential cross section"""

rundifcrosstest = 0
if rundifcrosstest == 1:
    def difcrosssection(wavelength,thetax,thetay,tau,Ee, crossingangle, obsangle):
        return (0.25*((4*(gamma(Ee)**2)*EL(wavelength)/(Egammaset(wavelength,Ee,crossingangle,obsangle)*(1 + (gamma(Ee)**2)*(thetax**2 + thetay**2)))) \
                   +  Egammaset(wavelength,Ee,crossingangle,obsangle)*(1 + (gamma(Ee)**2)*(thetax**2 + thetay**2))/(4*gamma(Ee)**2*EL(wavelength))) \
            - (1+Pt)*(np.cos(tau - phif(thetax,thetay))**2)*gamma(Ee)**2*(thetax**2 + thetay**2) \
                /(1+gamma(Ee)**2*(thetax**2 + thetay**2))**2)*8*np.pi*re**2*(Egammaset(wavelength, Ee, crossingangle, obsangle)/(4*gamma(Ee)*EL(wavelength)))**2
    ii = 100000
    testanglearray = np.linspace(start = -np.pi, stop = np.pi, num = ii)
    testdifcrosssection = np.zeros([ii,1])
    for n in range (ii):
        testdifcrosssection[n] = difcrosssection(wavelength_, testanglearray[n], 0, tau, Ee_/10000, 0, testanglearray[n])
    r = testdifcrosssection
    
    plt.axes(projection = 'polar')
    #plt.title(r'Differential cross section [m$^2$], 800 nm photon, Electron energy 250 MeV')
    plt.rcParams.update({'font.size': 12})
    plt.title(r'Differential cross section $d\sigma / d\Omega$ [m$^2$], $\lambda$ = 800 nm, E$_e$ = 25 keV')
    xT=plt.xticks()[0]
    xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$',r'$-\frac{\pi}{4}$']
    plt.xticks(xT, xL)
    #plt.set_rlabel_position(-22.5)
    plt.plot(testanglearray, testdifcrosssection)

rundifcrossenergy = 0
if rundifcrossenergy == 1:
    def X(Ee,wavelength):
        return 2*gamma(Ee)*EL(wavelength)*(1+Beta(Ee))/(me)
    
    #def Y(Ee,wavelength,obsangle):
        #return 2*gamma(Ee)*EL(wavelength)*(1-Beta(Ee)*math.cos(obsangle))/(me)
    
    def Y(Ee,wavelength,obsangle):
        return X(Ee,wavelength)*(Ee - Egammaset(wavelength,Ee,0,obsangle))/(Ee - EL(wavelength))
    
    def Yenergy(Ee,wavelength,Egamma):
        return X(Ee,wavelength)*(Ee - Egamma)/(Ee - EL(wavelength))
    
    def difcrossenergy(Ee,wavelength,obsangle):
        return 8*np.pi*re**2 *((X(Ee,wavelength)**-1 - Y(Ee,wavelength,obsangle)**-1)**2 + X(Ee,wavelength)**-1 - Y(Ee,wavelength,obsangle)**-1 \
                               + 0.25*(X(Ee,wavelength)/Y(Ee,wavelength,obsangle) + Y(Ee,wavelength,obsangle)/X(Ee,wavelength)))/(X(Ee,wavelength)*(Ee - EL(wavelength)))
            
    def difcrossenergy2(Ee,wavelength,Egamma):
        return 8*np.pi*re**2 *((X(Ee,wavelength)**-1 - Yenergy(Ee,wavelength,Egamma)**-1)**2 + X(Ee,wavelength)**-1 - Yenergy(Ee,wavelength,Egamma)**-1 \
                               + 0.25*(X(Ee,wavelength)/Yenergy(Ee,wavelength,Egamma) + Yenergy(Ee,wavelength,Egamma)/X(Ee,wavelength)))/(X(Ee,wavelength)*(Ee - EL(wavelength)))
    
    ii = 100    
    testanglearray = np.linspace(start = 0*math.pi, stop = 2*math.pi, num = ii)
    testenergyarray = np.linspace(start = 0, stop = Egammaset(wavelength_,Ee_,0,0), num = ii)
    testdifcrossenergy = np.zeros([ii,1])
    testdifcrossenergy2 = np.zeros([ii,1])
    for n in range (ii):
        testdifcrossenergy[n] = difcrossenergy(Ee_,wavelength_,testanglearray[n])
        testdifcrossenergy2[n] = difcrossenergy2(Ee_,wavelength_,testenergyarray[n])
    #plt.yscale('log')
    #maxvalue = testdifcrossenergy2.max()
    plt.plot(testenergyarray/testenergyarray.max(),testdifcrossenergy2/testdifcrossenergy2.max())
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Normalised scattered photon energy')
    plt.ylabel(r'Normalised d$\sigma$/dE$_\gamma$')
    

if runmontecarlo == 1:
    print("Monte Carlo integration starting")
    @jit
    def k_list(wavelength,dk,NMC):
        return np.random.uniform(kmin(wavelength,dk),kmax(wavelength,dk),NMC)
    @jit
    def theta_list(wavelength,Egamma,NMC):
        return np.random.uniform(-thetaymax(wavelength, Egamma), thetaymax(wavelength, Egamma), (NMC,2))
    @jit
    def xy_list(L,collimationangle,NMC):
        return np.random.uniform(-R(L,collimationangle), R(L,collimationangle), (NMC,2))
    @jit
    def domain_circle(x,max_value):
        return (np.power(x,2).sum())**0.5 < max_value
    @jit(parallel = True)
    def montecarloINTterm(Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, tau, dEe, dk, collimationangle,NMC):
        k_random_list = k_list(wavelength, dk, NMC)
        theta_random_list = theta_list(wavelength, Egamma, NMC)
        xy_random_list = xy_list(L, collimationangle, NMC)
        
        inside_domain_theta = [domain_circle(ii, thetaymax(wavelength, Egamma)) for ii in theta_random_list]
        inside_domain_xy = [domain_circle(ii,R(L,collimationangle)) for ii in xy_random_list]
        
        #frac_domain_theta = np.sum(inside_domain_theta)/len(inside_domain_theta)
        frac_domain_theta = inside_domain_theta.count(True)/len(inside_domain_theta)
        #frac_domain_xy = np.sum(inside_domain_xy)/len(inside_domain_xy)
        frac_domain_xy = inside_domain_xy.count(True)/len(inside_domain_xy)
        
        domain_theta = (2*thetaymax(wavelength, Egamma))**2 * frac_domain_theta
        domain_xy = (2*R(L,collimationangle))**2 * frac_domain_xy
        domain_k = (kmax(wavelength, dk) - kmin(wavelength, dk))
        
        total_domain = domain_theta * domain_xy * domain_k
        
        ii = 0
        #values = np.zeros(np.size(k_random_list))
        values = np.zeros(len(k_random_list))
        while ii < len(k_random_list):
        #for ii in inside_domain_theta:
            if(inside_domain_theta[ii] & inside_domain_xy[ii]):
                values[ii] = intterm(k_random_list[ii], Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, theta_random_list[ii,0], theta_random_list[ii,1], tau, xy_random_list[ii,0], xy_random_list[ii,1], dEe, dk)
            ii = ii+1
        if np.count_nonzero(values) !=0: 
            values_mean = values.sum()/np.count_nonzero(values)
            
        else:
            values_mean = 0
        
        integ = values_mean * total_domain
        
        return integ, frac_domain_theta, frac_domain_xy
    #montecarlotestvalue = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammatest, tau, dEe_, dk_, collimationangle_,NMC)
    
    INTvalue = np.zeros(np.size(Egammarange))
    for x in range(npts_):
        #process = multiprocessing.Process(target = montecarloINTterm, args =(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC))
        INTvalue[x], frac_domain_theta_test, frac_domain_xy_test = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC)
    
    
    
    print("Spectral Density calculated")
    if Energyrange == 'keV':
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phkeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-3,Specden)
        factor = 10**-3
    elif Energyrange == 'MeV':
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phMeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-6,Specden)
        factor = 10**-6
    else:
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue
        #plt.plot(Egammarange,Specden)
    if plotandsave == 1:
        plt.plot(factor*Egammarange,Specden)
        xlabelset = 'Energy (' + Energyrange + ')'
        plt.xlabel(xlabelset)
        plt.ylabel('Spectral Density')
        os.chdir('c:\\Users\\admorris\\Documents\\ICS Code\\Dump')
        plt.savefig(Case + 'Specdenplotterm4test.png')
        EnergySpecden = np.stack((Egammarange,Specden),axis=1)
        
        filename = (Case + "Specdensaveterm4test.txt")
        np.savetxt(filename,(EnergySpecden), delimiter = ',')
    
if runmontecarlotest == 1:
    print("Monte Carlo integration starting")
    @jit
    def k_list(wavelength,dk,NMC):
        return np.random.uniform(kmin(wavelength,dk),kmax(wavelength,dk),NMC)
    @jit
    def theta_list(wavelength,Egamma,NMC):
        y_list = np.random.uniform(-thetaymax(wavelength, Egamma), thetaymax(wavelength, Egamma), (NMC,1))
        x_list = np.zeros_like(y_list)
        for ii in range(NMC):
            x_list[ii] = np.random.uniform(-thetaxmax(wavelength,Egamma,y_list[ii]),thetaxmax(wavelength,Egamma,y_list[ii]))
        return y_list, x_list
    @jit
    def xy_list(L,collimationangle,NMC):
        x_list = np.random.uniform(-R(L,collimationangle), R(L,collimationangle), (NMC,1))
        y_list = np.zeros_like(x_list)
        for ii in range(NMC):
            y_list[ii] = np.random.uniform(-ytheta(L,collimationangle,x_list[ii]),ytheta(L,collimationangle,x_list[ii]))
        return y_list, x_list
    @jit
    def domain_circle(x,max_value):
        return (np.power(x,2).sum())**0.5 < max_value
    @jit
    def montecarloINTterm(Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, tau, dEe, dk, collimationangle,NMC):
        k_random_list = k_list(wavelength, dk, NMC)
        thetay_random_list, thetax_random_list = theta_list(wavelength, Egamma, NMC)
        y_random_list, x_random_list = xy_list(L, collimationangle, NMC)
        
        domain_theta = (2*thetaymax(wavelength, Egamma))**2
        domain_xy = (2*R(L,collimationangle))**2
        domain_k = (kmax(wavelength, dk) - kmin(wavelength, dk))
        
        total_domain = domain_theta * domain_xy * domain_k
        
        ii = 0
        #values = np.zeros(np.size(k_random_list))
        values = np.zeros(len(k_random_list))
        while ii < len(k_random_list):
        #for ii in inside_domain_theta:
            values[ii] = intterm(k_random_list[ii], Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax_random_list[ii], thetay_random_list[ii], tau, x_random_list[ii], y_random_list[ii], dEe, dk)
            ii = ii+1
        if np.count_nonzero(values) !=0: 
            values_mean = values.sum()/np.count_nonzero(values)
            
        else:
            values_mean = 0
        
        integ = values_mean * total_domain
        
        return integ
    #montecarlotestvalue = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammatest, tau, dEe_, dk_, collimationangle_,NMC)
    
    INTvaluetest = np.zeros(np.size(Egammarange))
    for x in range(npts_):
        #process = multiprocessing.Process(target = montecarloINTterm, args =(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC))
        INTvaluetest[x] = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC)
    
    
    
    print("Spectral Density calculated")
    if Energyrange == 'keV':
        Specdentest = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvaluetest*phkeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-3,Specdentest)
    elif Energyrange == 'MeV':
        Specdentest = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvaluetest*phMeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-6,Specdentest)
    else:
        Specdentest = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvaluetest
        #plt.plot(Egammarange,Specdentest)
    #xlabelset = 'Energy (' + Energyrange + ')'
    #plt.xlabel(xlabelset)
    #plt.ylabel('Spectral Density')
    #os.chdir('c:\\Users\\admorris\\Documents\\ICS Code\\Dump')
    #plt.savefig(Case + 'Specdenplot.png')
    #EnergySpecden = np.stack((Egammarange,Specden),axis=1)
    
    #filename = (Case + "Specdensavetest.txt")
    #np.savetxt(filename,(EnergySpecden), delimiter = ',')
if runmontecarlo & runmontecarlotest:
    plt.plot(factor*Egammarange,Specden, label = 'Test if in domain method')
    plt.plot(factor*Egammarange,Specdentest, label = 'Define varibles with each other method')
    plt.legend(loc = 'upper left')
    xlabelset = 'Energy (' + Energyrange + ')'
    plt.xlabel(xlabelset)
    plt.ylabel('Spectral Density')

if runqmc == 1:
    print("Quasi-Monte Carlo integration")
    INTmethod = "Quasi_Monte_Carlo"
    @jit
    def domain_circle(x,max_value):
        return (np.power(x,2).sum())**0.5 < max_value
    @jit(parallel = True)
    def quasimontecarloINTterm(Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, tau, dEe, dk, collimationangle,NQMC):
        
        sampler = qmc.Sobol(d=5)
        npsample = np.array(sampler.random_base2(m=QNMC))
        #npsample = np.array(sample)
        k_quasi_list = kmin(wavelength, dk)+ (kmax(wavelength,dk) - kmin(wavelength, dk))*npsample[:,0]
        theta_quasi_list = -thetaymax(wavelength, Egamma)+ (2*thetaymax(wavelength, Egamma))*npsample[:,1:3]
        xy_quasi_list = -R(L,collimationangle) + 2*R(L,collimationangle)* npsample[:,3:5]
        
        inside_domain_theta = [domain_circle(ii, thetaymax(wavelength, Egamma)) for ii in theta_quasi_list]
        inside_domain_xy = [domain_circle(ii,R(L,collimationangle)) for ii in xy_quasi_list]
        
        #frac_domain_theta = np.sum(inside_domain_theta)/len(inside_domain_theta)
        frac_domain_theta = inside_domain_theta.count(True)/len(inside_domain_theta)
        #frac_domain_xy = np.sum(inside_domain_xy)/len(inside_domain_xy)
        frac_domain_xy = inside_domain_xy.count(True)/len(inside_domain_xy)
        
        domain_theta = (2*thetaymax(wavelength, Egamma))**2 * frac_domain_theta
        domain_xy = (2*R(L,collimationangle))**2 * frac_domain_xy
        domain_k = (kmax(wavelength, dk) - kmin(wavelength, dk))
        
        total_domain = domain_theta * domain_xy * domain_k
        
        ii = 0
        #values = np.zeros(np.size(k_random_list))
        values = np.zeros(len(k_quasi_list))
        while ii < len(k_quasi_list):
        #for ii in inside_domain_theta:
            if(inside_domain_theta[ii] & inside_domain_xy[ii]):
                values[ii] = intterm(k_quasi_list[ii], Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, theta_quasi_list[ii,0], theta_quasi_list[ii,1], tau, xy_quasi_list[ii,0], xy_quasi_list[ii,1], dEe, dk)
            ii = ii+1
        if np.count_nonzero(values) !=0: 
            values_mean = values.sum()/np.count_nonzero(values)
            
        else:
            values_mean = 0
        
        integ = values_mean * total_domain
        
        return integ, frac_domain_theta, frac_domain_xy
    #montecarlotestvalue = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammatest, tau, dEe_, dk_, collimationangle_,NMC)
    
    INTvalue = np.zeros(np.size(Egammarange))
    for x in range(npts_):
        #process = multiprocessing.Process(target = montecarloINTterm, args =(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC))
        INTvalue[x], frac_domain_theta_test, frac_domain_xy_test = quasimontecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,QNMC)
    
    
    
    print("Spectral Density calculated")
    if Energyrange == 'keV':
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phkeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-3,Specden)
        factor = 10**-3
    elif Energyrange == 'MeV':
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phMeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-6,Specden)
        factor = 10**-6
    else:
        Specden = prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue
        #plt.plot(Egammarange,Specden)
    if plotandsave == 1:
        plt.plot(factor*Egammarange,Specden)
        xlabelset = 'Energy (' + Energyrange + ')'
        plt.xlabel(xlabelset)
        plt.ylabel('Spectral Density')
        os.chdir('c:\\Users\\admorris\\Documents\\ICS Code\\Dump\\')
        plt.savefig(Case + 'Specdenplotterm2.png')
        EnergySpecden = np.stack((Egammarange,Specden),axis=1)
        
        filename = (Case + "Specdensavetest.txt")
        np.savetxt(filename,(EnergySpecden), delimiter = ',')
    
    
endtime = time.time()
elapsed_time = -starttime + endtime
print('Execution time:', elapsed_time/60, 'minutes')