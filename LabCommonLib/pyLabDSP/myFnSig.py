# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:10:08 2017

@author: Chng Eng Siong
Date: Dec 2017, using python to solve DSP problems
purpose - to remove dependency on Matlab
This is myFnSig module => my module collecting functions supporting the DSP course
"""

import numpy as np
from scipy import signal

# our basic array types are numpy formats of array


# The following function generates a continuous time sinusoid
# given the amplitude A, F (cycles/seconds), Fs=sampling rate, start and endtime
def fnGenSampledSinusoid(A,Freq,Phi, Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    n = np.arange(sTime,eTime,1.0/Fs)
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]


def fnGenSampledSignalCosine(A,small_omega,Phi, numSamples):
    # Showing off how to use numerical python library to create arange
    n = np.arange(0,numSamples)
    yn = A*np.cos((small_omega*n)+Phi)
    return [n,yn]


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))



def fnGenDTComplexExp(A, w1, Phi, numSamples):
    n = np.arange(0,numSamples,1)
    y = np.multiply(np.power(A,n), np.exp(1j*(w1*n+Phi)))
    return [n,y]



def  fnConvolve(sampleX,impulseH):
  opY = np.zeros(len(sampleX)+len(impulseH)+1)
  if (len(sampleX)>= len(impulseH)):
      for j in range(len(impulseH)):
          stIdx  = j
          endIdx = j+len(sampleX)
          opY[stIdx:endIdx] = opY[stIdx:endIdx] +impulseH[j]*sampleX[0:len(sampleX)]

  return opY


def fnGenImpulseResponse(num, den, numSamples):
    n = np.arange(0,numSamples,1)
    impulseH = np.zeros(numSamples)
    impulseH[0] = 1
    y = signal.lfilter(num, den, impulseH)
    return [n,y]


def fn_lfilter(B,A,X):
    numNum = len(B)
    numDen = len(A)

    if A[0] != 1:
        print('expecting A[0] = 1 in fn_lfilter')
        raise Exception('Incorrect data')

    memX = np.zeros(numNum)
    B_np  = np.array(B)
    if numDen>=2:
        memY = np.zeros(numDen-1)
        A_np = np.array(A[1:])   # copying only second element onwards
        A_np = -1*A_np           # ensure the A coeffs sign is flip

    y = np.zeros(len(X))

    numElemX = len(X)
    for i in np.arange(numElemX):
       # doing the left side of DF1 structure
        memX[0] = X[i]   #rolling in the memory X input
        vec_left_op = np.multiply(memX,B_np)
        y[i] = np.sum(vec_left_op)

        memX = np.roll(memX, 1)  # getting ready for the next step
        memX[0] = 0  # we use roll, so circular shift, lets 0 shifted in element 0

        if numDen >= 2:
            vec_right_op = np.multiply(memY,A_np)
            sum_vec_right = np.sum(vec_right_op)
            y[i] = y[i] + sum_vec_right
            memY = np.roll(memY, 1)
            memY[0] = y[i]

    return y




# my unit testing for module myFnSig.py
if __name__ == "__main__":
    Amp=0.5; Freq=2000; Phi = 0; Fs=16000; sTime=0; eTime=1.0;
    [t,y]=fnGenSampledSinusoid(Amp,Freq,Phi,Fs,sTime,eTime)
