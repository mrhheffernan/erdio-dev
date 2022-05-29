import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: librosa dependencies apparently require specific versions of numpy, try numpy==1.21.4
import librosa
import librosa.display
import os

import scipy
from joblib import Parallel, delayed


# Function to load wav file with librosa
def load_data(index):
    fileprefix = 'futuramaErdos-'
    filesuffix = '.wav'
    fileindex = str(index)
    fileindex = fileindex.zfill(2)
    filepath = f'large_data/FuturamaValidation5s/'
    return librosa.load(filepath+fileprefix+fileindex+filesuffix,sr=None)


def write_csv_line(filenm, eq_cutoffs, row_ind):
    file = open(filenm+"_"+str(os.getpid())+".csv",'a+')

    y,sr = load_data(row_ind)
    ## get timing info
    t = np.round(librosa.get_duration(y=y,sr=sr),3)
    dt = t/len(y)
    k = np.fft.fftfreq(len(y), d=dt)

    ## hilbert transform filtering for "room noise"
    hil = np.abs(scipy.signal.hilbert(y))
    hilk = np.fft.fft(hil)
    hilk *= np.exp(-k*k / (2*10**2))
    hil2 = np.fft.ifft(hilk)
    hil2 = np.abs(hil2)
    hil2 *= hil.max() / hil2.max()
    filt_y = y * hil2/np.sqrt((y*y).mean())
    filt_y *= y.max() / filt_y.max()
    y_sq = filt_y*filt_y
    ## get crest factor
    Cr = filt_y.max() / np.sqrt(y_sq.mean())

    yk = np.fft.fft(filt_y)
    mag_yk = np.abs(yk)
    mag_yk = mag_yk[:len(mag_yk)//2]

    ## seperate into harmonic and percussive components
    D = librosa.stft(filt_y / filt_y.max())
    y_harmonic, y_percussive = librosa.decompose.hpss(D,margin=16.0)
    y_p = librosa.istft(y_percussive, length=len(filt_y))
    y_h = librosa.istft(y_harmonic, length=len(filt_y))
    Ptot = (filt_y**2).mean()
    ynorm = filt_y / np.sqrt(Ptot)
    Pnorm = (ynorm**2).mean()
    P_p = (y_p**2).mean()
    P_h = (y_h**2).mean()
    Anorm = np.sqrt(Pnorm/(P_h+P_p))
    y_p *= Anorm
    y_h *= Anorm

    y_percussive = librosa.stft(y_p)
    y_harmonic = librosa.stft(y_h)

    ## generate the MIDI range and corresponding frequencies
    st_k = np.fft.fftfreq(np.size(y_harmonic,0), d=dt)
    fbins = []
    keypitch = []
    midi = []
    maxlen=np.size(y_harmonic,0)
    p = -69
    kmin = 440*2**((p - 0.5)/12)
    kmax = 440*2**((p+0.5)/12)
    while kmax <= 20000:
        kmin = 440*2**((p - 0.5)/12)
        kmax = 440*2**((p+0.5)/12)
        keypitch.append(440*2**(p/12))
        midi.append(p+69)
        p += 1

    ## sum up the number of 3rds, 5ths, and chords
    harmonicality = 0
    for tm in range(np.size(y_harmonic,1)):
        p = -69
        kmin = 440*2**((p - 0.5)/12)
        kmax = 440*2**((p+0.5)/12)
        while kmax <= 20000:
            kmin = 440*2**((p - 0.5)/12)
            kmax = 440*2**((p+0.5)/12)
            if (len(np.abs(y_harmonic[(st_k[:maxlen] <= kmax) & (st_k[:maxlen] > kmin),tm])) != 0) :
                sumbin = np.abs(y_harmonic[(st_k[:maxlen] <= kmax) & (st_k[:maxlen] > kmin),tm]).sum() / len(np.abs(y_harmonic[(st_k[:maxlen] <= kmax) & (st_k[:maxlen] > kmin),tm]))
            else:
                sumbin = 0
            fbins.append(sumbin)
            p += 1

        peaks = librosa.util.localmax(np.asarray(fbins))

        maj3 = []
        min3 = []
        maj5 = []
        majchord = []
        fbins2 = np.asarray(fbins)
        if len(fbins2[fbins2 > 0] > 0):
            favg = fbins2[fbins2 > 0].max()
        else:
            favg = 1
        for i in range(len(fbins)-4):
            maj3.append(peaks[i]*peaks[i+4]*(fbins[i] + fbins[i+4])/favg)
        for i in range(len(fbins)-3):
            min3.append(peaks[i]*peaks[i+3]*(fbins[i] + fbins[i+3])/favg)
        for i in range(len(fbins)-7):
            maj5.append(peaks[i]*peaks[i+7]*(fbins[i] + fbins[i+7])/favg)
            majchord.append(peaks[i]*peaks[i+4]*peaks[i+7]*(fbins[i] + fbins[i+4] + fbins[i+7])/favg)


        maj3 = np.asarray(maj3)
        min3 = np.asarray(min3)
        maj5 = np.asarray(maj5)
        majchord = np.asarray(majchord)
        maj3 = len(maj3[np.log10(maj3 + 1e-12)>=-0.5])
        min3 = len(min3[np.log10(min3 + 1e-12)>=-0.5])
        maj5 = len(maj5[np.log10(maj5 + 1e-12)>=-0.5])
        majchord = len(majchord[np.log10(majchord + 1e-12)>=-0.5])
        harmonicality += maj3 + min3 + maj5 + majchord

    ## divide by the number of time slices to get mean harmonicality
    harmrate = harmonicality / np.size(y_harmonic,1)


    ## from the percussive component get the onsets for number of percussive hits
    y_out = librosa.istft(y_percussive, length=len(filt_y))

    onset_env = librosa.onset.onset_strength(y=y_out, sr=sr,
                                             max_size=5,
                                              aggregate=np.median)

    perc_peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=0)


    perc_rate = len(perc_peaks) #/ t

    ## generate equilizer values from fourier transform
    equilizer = np.zeros(len(eq_cutoffs))
    for i in range(len(equilizer)):
        index = 0
        num = 0
        while ((index < len(mag_yk)) & (k[index] < eq_cutoffs[i])):
            equilizer[i] += mag_yk[index]
            index += 1
            num += 1
        equilizer[i] /= num

    ## write everything to csv
    file.write(row_ind)
    for i in range(len(eq_cutoffs)):
        file.write(","+str(equilizer[i]))

    file.write(","+str(Cr)+","+str(P_h)+","+str(P_p)+","+str(harmrate)+","+str(perc_rate)+"\n")

    print("preprocessed audio file ", str(row_ind))
    file.close()


def make_csv(filenm):
    file = open(filenm+"_header.csv",'a+')
    file.write("index,eq_0,eq_10,eq_20,eq_30,eq_40,eq_60,eq_80,eq_120,eq_160,eq_230,eq_300,eq_450,eq_600,eq_900,eq_1200,eq_1800,eq_2400,eq_3700,eq_5000,eq_7500,eq_10000,eq_15000,eq_20000,crestfactor,harmonic_power,percussive_power,harmonic_hits,percussive_hits\n")
    file.close()

    eq_cutoffs = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 230.0, 300.0, 450.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0, 3700.0, 5000.0, 7500.0, 10000.0, 15000.0, 20000.0, np.inf]

    backend = 'loky'

    Parallel(n_jobs=40, backend = backend)(delayed(write_csv_line)(filenm, eq_cutoffs, row_ind) for row_ind in range(1,261))

make_csv('./large_data/ProcessedFuturamaValidation')
