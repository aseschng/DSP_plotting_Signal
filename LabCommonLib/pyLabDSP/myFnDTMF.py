# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:10:08 2017

@author: Chng Eng Siong
Date: Dec 2017, using python to solve DSP problems
This is myFnDTMF module => my module to generate DTMF signal
"""

import numpy as np

# The following function generates a 0.2seconds continuous time sinusoid
# of DTMF key 0..9, # and * keys
# for a 16KHz sampling rate
# FreqY = 697,770,852,941
# FreqX = 1209,1336,1477,1633

# returns the two freqeuncies of a DTMF key
def fnGet_F1F2_DTMF(whichkey):
    LookUpDictDTMF = {
                  '1':(697,1209), '2':(697,1336), '3':(697,1477),
                  '4':(770,1209), '5':(770,1336), '6':(770,1477),
                  '7':(852,1209), '8':(852,1336), '9':(852,1477),
                  '*':(941,1209), '0':(941,1336), '#':(941,1477),
                      }

    (F1,F2)=LookUpDictDTMF[whichkey]
    return((F1,F2))

# generate a discrete time sequence of DTMF whichKey for sampling rate Fs with duration =durTone(seconds)
def fnGenSampledDTMF(whichKey,Fs,durTone):
    (F1,F2) = fnGet_F1F2_DTMF(whichKey)
    Phi     = 0
    t       = np.arange(0,durTone,1.0/Fs)
    y       = np.cos(2 * np.pi * F1 * t + Phi) + np.cos(2 * np.pi * F2 * t + Phi)
    return [t,y]


def fnGenSampledDTMF_string(myTestKey,Fs,durTone):
    opy = []
    opt = []
    tmpTime = 0.0
    durSil  = 0.01;
    for c in myTestKey:
        [t, y] = fnGenSampledDTMF(c, Fs,  durTone)
        tmpTime = tmpTime+durTone+durSil
        y = 0.25 * y
        opy = np.concatenate((opy, y))
        silY = np.zeros( int(durSil*Fs))
        opy = np.concatenate((opy, silY))

    # generating the time index of the opy
    opt = np.linspace(0, tmpTime, opy.size, endpoint=True)
    return((opt,opy))

# my unit testing for module myFnSig.py
if __name__ == "__main__":
    Amp=1; Freq=2000; Phi = 0; Fs=16000; sTime=0; eTime=1.0;
    [t,y]=fnGenSampledDTMF('0',16000,0.5,1.5)
