# Iterative Adaptive Inverse Filtering with Glottal Flow Model:
# O. Perrotin and I. V. McLoughlin (2019)
#     "A spectral glottal flow model for source-filter separation of
#      speech", in IEEE International Conference on Acoustics, Speech, and
#      Signal Processing (ICASSP), Brighton, UK, May 12-17, pp. 7160-7164.

# Python translation of gfmiaif.m:
# https://github.com/operrotin/GFM-IAIF/tree/master



import numpy as np
import librosa
from scipy.signal import lfilter

def gfm_iaif(x, n_vt=48, n_gl=3, d=0.99, win=None):
    if win is None:
        win = np.hanning(x.size)
    
    # Addition of pre-frame
    lpf = n_vt+1
    x_preframe = np.append(np.linspace(-x[0], x[0], lpf), x)
    
    # cancel lip radiation contribution
    
    # lip radiation filter
    lip_coeffs = np.array([1, -d])
    
    # Gross glottis estimation
    gv_preframe = lfilter([1], lip_coeffs, x_preframe)
    gv = lfilter([1], lip_coeffs, x)
    
    glott_coeffs_gross = librosa.lpc(gv * win, order=1)
    
    for i in range(n_gl-1):
        # Cancel current estimate of glottis contribution
        v1x_preframe = lfilter(glott_coeffs_gross, 1, gv_preframe)
        v1x = v1x_preframe[lpf:]
        
        glott_coeffs_gross_x = librosa.lpc(v1x * win, order=1)
        glott_coeffs_gross = np.convolve(glott_coeffs_gross, glott_coeffs_gross_x)


    # Gross vocal tract estimation
    
    # cancel gross glottis contribution from speech signal
    v1_preframe = lfilter(glott_coeffs_gross, [1], gv_preframe)
    v1 = v1_preframe[lpf:]
    
    vt_coeffs_gross = librosa.lpc(v1 * win, order=n_vt)

    # Fine glottis estimation
    
    # cancel gross vocal tract contribution from speech signal
    g1_preframe = lfilter(vt_coeffs_gross, [1], gv_preframe)
    g1 = g1_preframe[lpf:]
    
    glott_coeffs_fine = librosa.lpc(g1 * win, order=n_gl)
    
    # Fine vocal tract estimation
    
    # cancel fine glottis contribution from speech signal
    v_preframe = lfilter(glott_coeffs_fine, [1], gv_preframe)
    v = v_preframe[lpf:]
    
    vt_coeffs_fine = librosa.lpc(v * win, order=n_vt)
    
    return vt_coeffs_fine, glott_coeffs_fine, lip_coeffs
