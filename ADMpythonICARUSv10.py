# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:47:03 2023

@author: admorris
Implementing a crossing angle into ICARUS
Lazy way, includes ACHG factor but not changing the energy properly, i.e. will scale down the maximum energy
"""

import math
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
import time
import datetime
#import multiprocessing
#import threading
from scipy.integrate import quad
#from mpmath import mp
import os
from scipy.stats import qmc
import sys
#from decimal import Decimal

""" Code Options """
#directoryname = 'c:\\Users\\alexd\\Documents\\University work\\Code\\ICARUS\\data'
directoryname = "c:\\Users\\admorris\\Documents\\ICS Code\\Data\\ICSsource\\3mmcol"
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
runmontecarlo = 1; # run the monte carlo integration 0 for no, 1 for yes independent variables but checks if both are within limits
runqmc = 0; # run quasi montecarlo integration, 0 for no, 1 for yes
plotandsave = 1; # plot graphs and save the files, 0 for no, 1 for yes
NMC = int(10e7); #number of total montecarlo integration points
QNMC = 20 # total number of quasi montecarlo points is 2^QNMC 27 for 134217728
Elowset = 6e6#manually set the lower limit for the energy range set to 0 for automatic lower limit
Ehighset = 0 #manually set the upper limit for the energy range set to 0 for automatic upper limit
Case = "ICSsource" # Case for integration, defines what variables to use
npts_ = 101 #number of energy steps to use for the integration
rmsBW = 10/100 # bandwidth setting for lowest energy
include_RACHG = 0
"""
Defining constants
"""
starttime = time.time();
now = datetime.datetime.now()
print("Start time is", now.time())
re = 2.8179403262e-15; #electron radius [m]
clight = 299792458; #speed of light [m/s]
hbar = 6.5821e-16; # reduced planck constant [eV.s]
me = 0.51099895e6; #mass of electron in eV
elecharge = 1.60217662e-19; # electron charge [C]
Thomsoncross = 6.65246e-29 # Thomson cross section
#mp.dps = 50
phMeVnC = 1e6 * 1e-9 / elecharge;
phkeVnC = 1e3 * 1e-9 / elecharge;

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
dispx=0
"""Case for variables"""

if Case == "FEBE250CASEB":
    Ee_ = 250e6
    Q_ = 100e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.070e-3
    crossingangledeg = 0
    crossingangle_ = crossingangledeg*np.pi/180 #in rad
    enx_ = 1e-6
    Betax_ = 1.23
    eny_ = 1e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 8e-5
    dk_ = 0.03185
    L_ = 10
    sigmaze_ = 2.4e-12 * clight
    sigmazL_ = 2e-12 * clight
    Energyrange = "MeV"
    print('Case is FEBE250CASEB')
elif Case == "CBETA150FWHM":
    Ee_ = 150e6
    Q_ = 32e-12
    Epulse_ = 62e-6
    wavelength_ = 1064e-9
    collimationangle_ = 0.256e-3
    crossingangledeg = 0
    crossingangle_ = crossingangledeg*np.pi/180 #in rad
    enx_ = 0.3e-6
    Betax_ = 0.269211
    eny_ = 0.3e-6
    Betay_ = 0.269211
    sigmaL_ = 25e-6
    dEe_ = 5e-4
    dk_ = 6.57e-4
    L_ = 10
    sigmaze_ = 3.33e-12 * clight
    sigmazL_ = 10e-12 * clight
    Energyrange = "keV"
    print("Case is CBETA150FWHM")
elif Case == "CBETA150testing":
    Ee_ = 150e6
    Q_ = 32e-12
    Epulse_ = 62e-6
    wavelength_ = 1064e-9
    collimationangle_ = 0.5e-3
    crossingangle_ = 0
    enx_ = 0.3e-6
    Betax_ = 0.01
    eny_ = 0.3e-6
    Betay_ = 0.01
    sigmaL_ = 25e-6
    dEe_ = 1*5e-4
    dk_ = 6.57e-4
    L_ = 10
    
    Energyrange = "keV"
    print("Case is CBETA150testing")
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
    
    Energyrange = "MeV"
    print('Case is ', Case)
    
elif Case == "FEBE250CASEBtesting":
    Ee_ = 250e6
    Q_ = 100e-12
    Epulse_ = 0.1
    wavelength_ = 800e-9
    collimationangle_ = 0.070e-3
    crossingangle_ = 0
    enx_ = 1.3e-6
    Betax_ = 1.23
    eny_ = 1.3e-6
    Betay_ = 1.23
    sigmaL_ = 45e-6
    dEe_ = 2*8e-5 #
    dk_ = 1*0.03185 #
    L_ = 10
    
    Energyrange = "MeV"
    print('Case is FEBE250CASEBtesting')
    
elif Case == "DIANA680":
    Ee_ = 680e6
    Q_ = 100e-12
    Epulse_ = 100e-6
    wavelength_ = 1064e-9
    collimationangle_ = 0.09461e-3
    crossingangle_ = 5*np.pi/180 #in rad
    enx_ = 0.375e-9
    Betax_ = 0.718
    eny_ = 0.375e-9
    Betay_ = 0.718
    sigmaL_ = 25e-6
    dEe_ = 5e-4
    dk_ = 6.57e-4
    L_ = 10
    sigmaze_ = 1e-3
    sigmazL_ = 5.7e-12 * clight
    Energyrange = "MeV"
    print('Case is DIANA680')
    
elif Case == "HELIOS2":
    Ee_ = 700e6
    Q_ = 0.83e-9 #68.98e-12
    Epulse_ = 100e-6
    wavelength_ = 1064e-9
    collimationangle_ = 60e-3
    crossingangle_ = 0*np.pi/180 #in rad
    crossingangledeg = crossingangle_*180/np.pi
    enx_ = 1193e-6
    Betax_ = 2.03#(1.5e-3)**2/ex(Ee_,enx_)#2.59#1.44611
    eny_ = 119.3e-6
    Betay_ = 1.45#(1.1e-3)**2/ey(Ee_,eny_)#2.03481
    sigmaL_ = 0.5e-3
    dEe_ = 0.075e-2
    dk_ = 0.05e-2
    L_ = 10
    sigmaze_ = 74.5e-12 * clight
    sigmazL_ = 20e-12 * clight
    Energyrange = "MeV"
    print('Case is HELIOS2')
    
elif Case == "HELIOSmod":
    Ee_ = 700e6
    Q_ = 0.89e-9 #68.98e-12
    Epulse_ = 1.8e-3
    wavelength_ = 1064e-9
    collimationangle_ = 0.1e-3
    crossingangle_ = 0*np.pi/180 #in rad
    crossingangledeg = 0
    enx_ = 1370.8658284914284*53.18e-9
    Betax_ = 0.96
    eny_ = 1370.8658284914284*52.06e-9
    Betay_ = 3.52
    sigmaL_ = 0.41e-3
    dEe_ = 9e-4
    dk_ = 0.05e-2
    L_ = 10
    sigmaze_ = 0.5e-3
    sigmazL_ = 20e-12 * clight
    dispx = 0.45
    dispy = -0.07
    Energyrange = "MeV"
    print('Case is HELIOSmod')
elif Case == "HELIOSmodnodisp":
    Ee_ = 700e6
    Q_ = 0.89e-9 #68.98e-12
    Epulse_ = 1.8e-3
    wavelength_ = 1064e-9
    collimationangle_ = 0.1e-3
    crossingangle_ = 0*np.pi/180 #in rad
    crossingangledeg = 0
    enx_ = 1370.8658284914284*53.18e-9
    Betax_ = 0.614
    eny_ = 1370.8658284914284*52.06e-9
    Betay_ = 3.56#(1.1e-3)**2/ey(Ee_,eny_)#2.03481
    sigmaL_ = 0.41e-3
    dEe_ = 9e-4
    dk_ = 0.05e-2
    L_ = 10
    sigmaze_ = 0.5e-3
    sigmazL_ = 20e-12 * clight
    dispx = 0#0.45
    dispy = 0#-0.07
    Energyrange = "MeV"
    print('Case is HELIOSmodnodisp')
elif Case == "HELIOSmodcol":
    Ee_ = 700e6
    Q_ = 0.83e-9 #68.98e-12
    Epulse_ = 167e-6
    wavelength_ = 1064e-9
    collimationangle_ = 0.1e-3
    crossingangle_ = 0*np.pi/180 #in rad
    crossingangledeg = 0
    enx_ = 1370.8658284914284*53.18e-9
    Betax_ = 0.96#(1.5e-3)**2/ex(Ee_,enx_)#2.59#1.44611
    eny_ = 1370.8658284914284*52.06e-9
    Betay_ = 3.52#(1.1e-3)**2/ey(Ee_,eny_)#2.03481
    sigmaL_ = 0.5e-3
    dEe_ = 8.7e-4
    dk_ = 0.05e-2
    L_ = 10
    sigmaze_ = 0.5e-3
    sigmazL_ = 20e-12 * clight
    dispx = 0.45
    dispy = -0.07
    Energyrange = "MeV"
    print('Case is HELIOSmod')
    
elif Case == "ICSsource":
    Ee_ = 700e6
    Q_ = 0.89e-9 #68.98e-12
    Epulse_ = 4.34e-3
    wavelength_ = 1064e-9
    collimationangle_ = 0.3e-3
    crossingangle_ = 0*np.pi/180 #in rad
    crossingangledeg = 0
    enx_ = 1370.8658284914284*53.18e-9
    Betax_ = 0.96#(1.5e-3)**2/ex(Ee_,enx_)#2.59#1.44611
    eny_ = 1370.8658284914284*52.06e-9
    Betay_ = 3.52#(1.1e-3)**2/ey(Ee_,eny_)#2.03481
    sigmaL_ = 0.45e-3
    dEe_ = 8.7e-4
    dk_ = 0.05e-2
    L_ = 10
    sigmaze_ = 0.45e-3
    sigmazL_ = 20e-12 * clight
    dispx = 0.45
    dispy = -0.07
    Energyrange = "MeV"
    print('Case is ICSsource')
else:
       
    print('Invalid case used')
    sys.exit()

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
"""Miyahara crossing angle factors implementation  """
@jit
def epsilonn(Ee, epsilon):
    return gamma(Ee)*epsilon

@jit
def sigmaelectroncrossing(BetaIP,Ee,epsilon, disp = 0, dMom = 0):
    return (BetaIP*epsilon/gamma(Ee) + (disp*dMom)**2)**0.5

@jit
def ZR(sigmaL, wavelength):
    return 4*np.pi*sigmaL**2 / wavelength

@jit
def H(crossing_angle,Ee, epsilonx, BetaIPx, epsilony, BetaIPy, sigmaze, sigmazL, sigmaL,disp,dEe):
    return np.cos(crossing_angle/2)*((sigmaelectroncrossing(BetaIPx,Ee,epsilonx,disp,dEe)**2 +sigmaL**2)* (sigmaelectroncrossing(BetaIPy,Ee,epsilony)**2 + sigmaL**2) / (np.pi * (sigmaze**2 + sigmazL**2)))**0.5

@jit
def U2j(epsilon, BetaIP, sigmaL, wavelength, Ee,disp,dEe):
    return (((sigmaelectroncrossing(BetaIP,Ee,epsilon,disp,dEe)**2/BetaIP**2) + (sigmaL**2 / ZR(sigmaL, wavelength)**2))/2)**0.5

@jit
def h(crossing_angle, epsilonx, Zc, BetaIPx, sigmaL, wavelength, sigmaze, sigmazL, Ee,disp,dEe):
    return np.sin(crossing_angle/2)**2 / (sigmaelectroncrossing(BetaIPx,Ee,epsilonx,disp,dEe)**2 + sigmaL**2 + U2j(epsilonx, BetaIPx, sigmaL, wavelength, Ee,disp,dEe)**2*Zc**2) \
        + np.cos(crossing_angle/2)**2 / (sigmaze**2 + sigmazL**2)

@jit        
def MiyaharaIntegrand(crossing_angle, BetaIPx, Ee, epsilonx,sigmaL, BetaIPy, epsilony, sigmaze, sigmazL, wavelength, Zc,disp,dEe):
    return H(crossing_angle,Ee, epsilonx, BetaIPx, epsilony, BetaIPy, sigmaze, sigmazL, sigmaL,disp,dEe)*np.exp(-h(crossing_angle, epsilonx, Zc, BetaIPx, sigmaL, wavelength, sigmaze, sigmazL, Ee,disp,dEe)* Zc**2)\
        / (((sigmaelectroncrossing(BetaIPx,Ee,epsilonx,disp,dEe)**2 + sigmaL**2 + U2j(epsilonx, BetaIPx, sigmaL, wavelength, Ee,disp,dEe)**2*Zc**2)**0.5) * ((sigmaelectroncrossing(BetaIPy,Ee,epsilony)**2 + sigmaL**2 + U2j(epsilony, BetaIPy, sigmaL, wavelength, Ee,disp,dEe)**2*Zc**2))**0.5)

        
def RACHG(crossing_angle, BetaIPx, Ee, epsilonx,sigmaL, BetaIPy, epsilony, sigmaze, sigmazL, wavelength, min_limit, max_limit,disp,dEe):
    return quad(lambda Zc: MiyaharaIntegrand(crossing_angle, BetaIPx, Ee, epsilonx,sigmaL, BetaIPy, epsilony, sigmaze, sigmazL, wavelength, Zc,disp,dEe), min_limit, max_limit)

def RAC(BetaIPx,epsilonx,Ee,sigmaxL,sigmaze,sigmazL,crossingangle,dispx = 0,dMomx=0):
    sigmax = (sigmaelectroncrossing(BetaIPx,Ee,epsilonx, dispx, dMomx)**2 + sigmaxL**2)**0.5
    sigmaz = (sigmaze**2 + sigmazL**2)**0.5
    return sigmax*np.cos(crossingangle/2)/((sigmax*np.cos(crossingangle/2))**2 + (sigmaz*np.sin(crossingangle/2))**2)**0.5

if include_RACHG == 1:

    RACHGreductionfactor = RACHG(crossingangle_, Betax_, Ee_, enx_, sigmaL_, Betay_, eny_, sigmaze_, sigmazL_, wavelength_, -0.1, 0.1,dispx,dEe_)
    RACfactor = RAC(Betax_,enx_,Ee_,sigmaL_,sigmaze_,sigmazL_,crossingangle_)
else:
    RACHGreductionfactor=[1]

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
def sigmaelectron(BetaIP,Ee,epsilon, disp = 0, dMom = 0):
    return (BetaIP*epsilonn(Ee,epsilon)/gamma(Ee) + (disp*dMom)**2)**0.5

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

"""Flux calculations """

def convxy(x,y):
    return (x**2 + y**2)**0.5

def Luminosity(Q,Epulse,wavelength,sigmaex,sigmaLx,sigmaey,sigmaLy):
    return Ne(Q)* NL(Epulse,wavelength)/(2*np.pi*convxy(sigmaex,sigmaLx)*convxy(sigmaey,sigmaLy))

def Comptoncross(Ee,crossingangle,wavelength):
    return Thomsoncross*(1 - 2*gamma(Ee)*EL(wavelength)*(1+Beta(Ee)*np.cos(crossingangle))/me)

def Flux(Ee,crossingangle,wavelength,Q,Epulse,sigmaex,sigmaLx,sigmaey,sigmaLy,reprate):
    return Comptoncross(Ee,crossingangle,wavelength)*Luminosity(Q,Epulse,wavelength,sigmaex,sigmaLx,sigmaey,sigmaLy)*reprate



"""Sun intermediatiries"""
"""
@jit  
def gammabar(wavelength,Egamma,thetax,thetay):
    return np.real(((2*Egamma*EL(wavelength))*((1+(me**2*(4*EL(wavelength) - Egamma*(thetax**2 + thetay**2))) \
            /(4*Egamma*EL(wavelength)**2))**0.5 + 1)) \
            /(me*(4*EL(wavelength) - Egamma*(thetax**2 + thetay**2))))
"""
"""
@jit
def gammabar(wavelength,Egamma,thetax,thetay,crossingangle, Ee):
    return (2*Egamma*EL(wavelength)/(me*(4*EL(wavelength) - Egamma*(thetax**2 + thetay**2))))* \
        (1+(1 + ((4*EL(wavelength)- Egamma*(thetax**2 + thetay**2))*me**2)/(4*Egamma*EL(wavelength)**2))**0.5)
"""        
@jit
def gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee):
    EL_ = EL(wavelength)
    return (2*Egamma*EL_/(me*(2*EL_*(1+ Beta(Ee)*np.cos(crossingangle)) - Egamma*(thetax**2 + thetay**2))))* \
        (1+(1 + ((2*EL_*(1+ Beta(Ee)*np.cos(crossingangle))- Egamma*(thetax**2 + thetay**2))*me**2)/(4*Egamma*EL_**2))**0.5)
       
@jit
def X(wavelength,Egamma,thetax,thetay, Ee, crossingangle):
    return 2*gammabar(wavelength,Egamma,thetax,thetay, crossingangle,Ee)*EL(wavelength)*(1+Beta(Ee)*np.cos(crossingangle))/(me)
@jit
def Y(wavelength,Egamma,thetax,thetay, Ee, crossingangle):
    return 2*gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)*Egamma*(1-Beta(Ee)*np.cos((thetax**2+thetay**2)**0.5))/(me)

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
def term2(wavelength,Egamma,thetax,thetay, crossingangle, Ee):
    gammabar_ = gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)
    return gammabar_/(1 + (2*gammabar_*EL(wavelength))/me)
@jit
def term3(wavelength,Egamma,thetax,thetay,tau, crossingangle, Ee):
    
    return 0.25*((4*(gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2)*EL(wavelength)/(Egamma*(1 + (gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2)*(thetax**2 + thetay**2)))) \
               +  Egamma*(1 + (gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2)*(thetax**2 + thetay**2))/(4*gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2*EL(wavelength))) \
        - (1+Pt)*(np.cos(tau - phif(thetax,thetay))**2)*gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2*(thetax**2 + thetay**2) \
            /(1+gammabar(wavelength,Egamma,thetax,thetay, crossingangle, Ee)**2*(thetax**2 + thetay**2))**2
@jit            
def term3crossing(wavelength,Egamma,thetax,thetay,tau, crossingangle, Ee):
     X_ = X(wavelength,Egamma,thetax,thetay, Ee, crossingangle)
     Y_ = Y(wavelength,Egamma,thetax,thetay, Ee, crossingangle)
     return 0.25*(X_/Y_ + Y_/X_) \
         + (1+Pt)*(np.cos(tau - phif(thetax,thetay))**2)*((1/X_ - 1/Y_)**2 \
            + 1/X_ - 1/Y_)

@jit
def term4(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL, crossingangle):
    return np.exp((-(((thetax - xd/L)**2)/(2*sigmathetax(Ee, enx, Betax, L, k, wavelength, sigmaL)**2)) \
                    - (((thetay - yd/L)**2)/(2*sigmathetay(Ee, eny, Betay, L, k, wavelength, sigmaL)**2)) \
                        - (((gammabar(wavelength, Egamma, thetax, thetay, crossingangle, Ee) - gamma(Ee))**2)/(2*sigmagamma(dEe, Ee)**2)) \
                            - (((k - klaser(wavelength))**2)/(2*sigmak(dk, wavelength)**2))))
        

@jit
def intterm(k, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax, thetay, tau, xd, yd, dEe, dk,crossingangle=0):
    return term1(k,Betax,Ee,enx,wavelength,Betay,eny,L,sigmaL)*term2(wavelength,Egamma,thetax,thetay,crossingangle,Ee)\
        *term3(wavelength,Egamma,thetax,thetay,tau,crossingangle,Ee)*term4(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL,crossingangle)

@jit
def inttermcrossing(k, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetax, thetay, tau, xd, yd, dEe, dk, crossingangle):
    return term1(k,Betax,Ee,enx,wavelength,Betay,eny,L,sigmaL)*term2(wavelength,Egamma,thetax,thetay, crossingangle, Ee)\
        *term3crossing(wavelength,Egamma,thetax,thetay,tau, crossingangle, Ee)*term4(thetax, xd, L, Ee, enx, Betax, k, wavelength, thetay, yd, eny, Betay, Egamma, dEe, dk, sigmaL, crossingangle)

@jit     
def Elow(rmsBW, wavelength, Ee, crossingangle):
    return (1 - rmsBW*3*(2*np.log(2))**0.5)*Egammaset(wavelength, Ee, crossingangle, 0)
@jit
def Ehigh(wavelength, Ee, dEe, crossingangle):
    return Egammaset(wavelength, Ee*(1+5*dEe), crossingangle, 0)
@jit
def Estep(wavelength, Ee, dEe, crossingangle, npts):
    return (Ehigh(wavelength, Ee, dEe, crossingangle) - Elow(rmsBW, wavelength, Ee, crossingangle))/npts


"""Creating an array of energies to loop through """

if Elowset != 0:
    if Ehighset !=0:
        Egammarange = np.linspace(start=Elowset, stop=Ehighset, num=npts_)
    else:
        Egammarange = np.linspace(start=Elowset, stop=Ehigh(wavelength_, Ee_, dEe_, crossingangle_), num=npts_)
else:
    if Ehighset ==0:
        Egammarange = np.linspace(start=Elow(rmsBW, wavelength_, Ee_, crossingangle_), stop=Ehigh(wavelength_, Ee_, dEe_, crossingangle_), num=npts_)
    else:
        Egammarange = np.linspace(start=Elow(rmsBW, wavelength_, Ee_, crossingangle_), stop = Ehighset, num = npts_)



    

if runmontecarlo == 1:
    print("Monte Carlo integration starting")

    @jit
    def domain_circle(x1,x2,max_value): # checks if the sum of squares is less than a max value
        return (x1**2+x2**2)**0.5 < max_value
    @jit
    def montecarloINTterm(Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, tau, dEe, dk, collimationangle,NMC, crossingangle):
        #generate the monte carlo integration points
        
        #defines the limits of the domain
        domain_theta = (2*thetaymax(wavelength, Egamma))**2
        domain_xy = (2*R(L,collimationangle))**2
        domain_k = (kmax(wavelength, dk) - kmin(wavelength, dk))
        
        total_domain = domain_theta * domain_xy * domain_k
        
        ii = 0
        count = 0
        
        #performs the integration on values which are in both domains
        #values = np.zeros(len(k_random_list))
        cum_vals = 0
        nan_count = 0
        while ii < NMC:
        #for ii in inside_domain_theta:
            krand = np.random.uniform(kmin(wavelength,dk),kmax(wavelength,dk))
            thetaxrand = np.random.uniform(-thetaymax(wavelength, Egamma), thetaymax(wavelength, Egamma))
            thetayrand = np.random.uniform(-thetaymax(wavelength, Egamma), thetaymax(wavelength, Egamma))
            xrand = np.random.uniform(-R(L,collimationangle), R(L,collimationangle))
            yrand = np.random.uniform(-R(L,collimationangle), R(L,collimationangle))
            inside_domain_theta = domain_circle(thetaxrand,thetayrand,thetaymax(wavelength, Egamma))
            inside_domain_xy = domain_circle(xrand,yrand, R(L,collimationangle))
            if(inside_domain_theta & inside_domain_xy):
                #values_temp = inttermcrossing(krand, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetaxrand, thetayrand, tau, xrand, yrand, dEe, dk, crossingangle)
                values_temp = intterm(krand, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, thetaxrand, thetayrand, tau, xrand, yrand, dEe, dk, crossingangle)
                if not np.isnan(values_temp):
                
                    cum_vals = cum_vals + values_temp
                    count = count + 1
                else:
                    nan_count = nan_count + 1
            ii = ii+1
        
            
        values_mean = cum_vals/NMC#values.sum()/np.count_nonzero(values)
        
        integ = values_mean * total_domain
        
        return integ, count, nan_count
    #montecarlotestvalue = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammatest, tau, dEe_, dk_, collimationangle_,NMC)
    
    INTvalue = np.zeros(np.size(Egammarange))
    counts = np.zeros(np.size(Egammarange))
    nan_count = np.zeros(np.size(Egammarange))
    #values_mean = np.zeros(np.size(Egammarange))
    
    for x in range(npts_):
        
        INTvalue[x],counts[x], nan_count[x] = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC, crossingangle_)
        
        print("Point", (x+1), "/", npts_)
    
    print("Spectral Density calculated")
    if Energyrange == 'keV':
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phkeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-3,Specden)
        factor = 10**-3
    elif Energyrange == 'MeV':
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phMeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-6,Specden)
        factor = 10**-6
    else:
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue
        #plt.plot(Egammarange,Specden)
    if plotandsave == 1:
        Egammarangeangled = Egammarange#*Egammaset(wavelength_, Ee_, crossingangle_, 0)/ Egammaset(wavelength_, Ee_, 0, 0)
        plt.plot(factor*Egammarangeangled,Specden)
        xlabelset = 'Energy (' + Energyrange + ')'
        plt.xlabel(xlabelset)
        plt.ylabel('Spectral Density [ph/(MeV nC)]')
        #plt.title(Case)
        os.chdir(directoryname) 
        filename = (Case + '_n' + str(npts_) + '_python_MC_'+ str(int(Egammarangeangled[0])) +'_'+ str(crossingangledeg) + 'deg')
        plt.savefig(filename + '.png', dpi= 300)
        EnergySpecden = np.stack((Egammarangeangled,Specden),axis=1)
        
        #filename = (Case + "n30.txt")
        np.savetxt(filename + '.txt',(EnergySpecden), delimiter = ',')
    


if runqmc == 1:
    print("Quasi-Monte Carlo integration")
    INTmethod = "Quasi_Monte_Carlo"
    @jit
    def domain_circle(x1,x2,max_value): # checks if the sum of squares is less than a max value
        return (x1**2+x2**2)**0.5 < max_value
    @jit
    def quasimontecarloINTterm(Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, tau, dEe, dk, collimationangle,NQMC,crossingangle):
        
        sampler = qmc.Sobol(d=5)
        npsample = np.array(sampler.random_base2(m=QNMC))
        #npsample = np.array(sample)
        #k_quasi_list = kmin(wavelength, dk)+ (kmax(wavelength,dk) - kmin(wavelength, dk))*npsample[:,0]
        #theta_quasi_list = -thetaymax(wavelength, Egamma)+ (2*thetaymax(wavelength, Egamma))*npsample[:,1:3]
        #xy_quasi_list = -R(L,collimationangle) + 2*R(L,collimationangle)* npsample[:,3:5]
        
        #inside_domain_theta = [domain_circle(ii, thetaymax(wavelength, Egamma)) for ii in theta_quasi_list]
        #inside_domain_xy = [domain_circle(ii,R(L,collimationangle)) for ii in xy_quasi_list]
        
        
        
        domain_theta = (2*thetaymax(wavelength, Egamma))**2 
        domain_xy = (2*R(L,collimationangle))**2 
        domain_k = (kmax(wavelength, dk) - kmin(wavelength, dk))
        
        total_domain = domain_theta * domain_xy * domain_k
        cum_vals = 0
        ii = 0
        total_points = len(npsample)
        #values = np.zeros(np.size(k_random_list))
        
        while ii < total_points:
            k_val = kmin(wavelength, dk)+ (kmax(wavelength,dk) - kmin(wavelength, dk))*npsample[ii,0]
            theta_vals = -thetaymax(wavelength, Egamma)+ (2*thetaymax(wavelength, Egamma))*npsample[ii,1:3]
            xy_vals = -R(L,collimationangle) + 2*R(L,collimationangle)* npsample[ii,3:5]
            inside_domain_theta = domain_circle(theta_vals[0],theta_vals[1],thetaymax(wavelength, Egamma))
            inside_domain_xy = domain_circle(xy_vals[0],xy_vals[1],R(L,collimationangle))
        #for ii in inside_domain_theta:
            if(inside_domain_theta & inside_domain_xy):
                
                values_temp = inttermcrossing(k_val, Betax, Ee, enx, wavelength, Betay, eny, L, sigmaL, Egamma, theta_vals[0], theta_vals[1], tau, xy_vals[0], xy_vals[1], dEe, dk,crossingangle)
                if not np.isnan(values_temp):
                    cum_vals = cum_vals+values_temp
            ii = ii+1
        
        values_mean = cum_vals/total_points
              
        integ = values_mean * total_domain
        
        return integ
    #montecarlotestvalue = montecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammatest, tau, dEe_, dk_, collimationangle_,NMC)
    
    INTvalue = np.zeros(np.size(Egammarange))
    for x in range(npts_):
        #process = multiprocessing.Process(target = montecarloINTterm, args =(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,NMC))
        INTvalue[x] = quasimontecarloINTterm(Betax_, Ee_, enx_, wavelength_, Betay_, eny_, L_, sigmaL_, Egammarange[x], tau, dEe_, dk_, collimationangle_,QNMC,crossingangle_)
        print("Point", (x+1), "/", npts_)
    
    
    print("Spectral Density calculated")
    if Energyrange == 'keV':
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phkeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-3,Specden)
        factor = 10**-3
    elif Energyrange == 'MeV':
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue*phMeVnC*elecharge/Q_
        #plt.plot(Egammarange*10**-6,Specden)
        factor = 10**-6
    else:
        Specden = RACHGreductionfactor[0]*prefactor(Q_, Epulse_, wavelength_, sigmaL_, dEe_, Ee_, dk_, L_)*INTvalue
        #plt.plot(Egammarange,Specden)
    if plotandsave == 1:
        plt.plot(factor*Egammarange,Specden)
        xlabelset = 'Energy (' + Energyrange + ')'
        plt.xlabel(xlabelset)
        ylabelset = 'Spectral Density [ph/('+ Energyrange +' nC)]'
        plt.ylabel(ylabelset)
        plt.title(Case)
        os.chdir(directoryname)
        filename = (Case + '_n' + str(npts_) + '_python_QMC_'+ str(int(Egammarange[0])) +'_'+ str(crossingangledeg) + 'deg')
        plt.savefig(filename + '.png')
        EnergySpecden = np.stack((Egammarange,Specden),axis=1)
        
        #filename = (Case + "n30.txt")
        np.savetxt(filename + '.txt',(EnergySpecden), delimiter = ',')
        
        """
        plt.plot(factor*Egammarange,Specden)
        xlabelset = 'Energy (' + Energyrange + ')'
        plt.xlabel(xlabelset)
        plt.ylabel('Spectral Density')
        #os.chdir('c:\\Users\\admorris\\Documents\\ICS Code\\Dump\\quasimontecarlo\\')
        plt.savefig(Case + 'Specdenplotgammabartest.png')
        EnergySpecden = np.stack((Egammarange,Specden),axis=1)
        
        filename = (Case + "Specdensavetestgammabartest.txt")
        np.savetxt(filename,(EnergySpecden), delimiter = ',')"""
    
    
endtime = time.time()
elapsed_time = -starttime + endtime
print('Execution time:', elapsed_time/60, 'minutes')


