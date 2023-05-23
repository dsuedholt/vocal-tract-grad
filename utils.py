import torch
import librosa
import scipy
from gfm_iaif import gfm_iaif
import numpy as np

def make_z(n_points):
    return torch.exp(1j * torch.linspace(0, torch.pi, n_points))

def to_log_mag(freq_response, rel_to_max=True, eps=1e-7):
    mag = torch.abs(freq_response)
    if rel_to_max:
        div = torch.max(mag)
    else:
        div = 1.0
    return 20 * torch.log10(mag / div + eps)

def weighted_log_mag_mse_loss(pred, target):
    nfft = (pred.numel() - 1) * 2
    fftfreqs = librosa.fft_frequencies(sr=48000, n_fft=nfft)
    amplitude_cweight = torch.tensor(librosa.db_to_power(librosa.C_weighting(fftfreqs)))
    pred = to_log_mag(pred * amplitude_cweight)
    target = to_log_mag(target * amplitude_cweight)
    return torch.mean(torch.square((pred - target)))# * amplitude_cweight))

def log_mag_mse_loss(pred, target):
    return torch.nn.functional.mse_loss(to_log_mag(pred), to_log_mag(target))

def decompose(audio, n_vt=44):
    vt_coeffs, gl_coeffs, lip_coeffs = gfm_iaif(audio, n_vt=n_vt)
    
    padded_audio = np.append(np.linspace(-audio[0], audio[0], n_vt+1), audio)
    glottal_waveform = scipy.signal.lfilter(vt_coeffs, [1], padded_audio)

    return glottal_waveform[n_vt+1:], vt_coeffs

def freqz(coeffs, n_points):
    _, fr = scipy.signal.freqz([1], coeffs, worN=n_points, include_nyquist=True)
    return fr

def yin(frame, sr, fmin=70, fmax=500):
    return librosa.yin(frame, fmin=fmin, fmax=fmax, frame_length=frame.size, hop_length=frame.size, sr=sr, center=False, trough_threshold=0.01)[0]

class MaxCorrLoss:
    def __init__(self, T):
        self.T = T

    def __call__(self, pred, target):
        lags = torch.arange(self.T, dtype=torch.int32)
        corrs = torch.zeros_like(lags)
        for i in lags:
            corrs[i.item()] = torch.sum(torch.roll(target, i.item()) * pred)

        shift = torch.argmax(corrs)
        target = torch.roll(target, shift.item())

        target = target / torch.max(target)
        pred = pred / torch.max(pred)

        return torch.nn.functional.mse_loss(pred, target)
        
def frames_to_samples(frames, frame_len, numpy=False):
    if numpy:
        frames = torch.tensor(frames)
    
    n_frames, n_dims = frames.shape
    if n_frames == 1:
        samples = torch.repeat_interleave(frames, frame_len, dim=0)
        if numpy:
            return samples.detach().numpy()
        return samples
    
    samples = torch.zeros((frame_len * (n_frames-1), n_dims))
    for i in range(1, n_frames):
        interp = torch.arange(frame_len) / frame_len
        prev = torch.repeat_interleave(torch.unsqueeze(frames[i-1], dim=0), frame_len, dim=0)
        curr = torch.repeat_interleave(torch.unsqueeze(frames[i], dim=0), frame_len, dim=0)
        samples[(i-1)*frame_len:i*frame_len] = ((1 - interp) * prev.T + interp * curr.T).T
    
    if numpy:
        return samples.detach().numpy()
    return samples

def h1h2(x, f0, sr):
    # calculate distance between magnitude of first and second harmonic in dB
    nfft = x.size
    x = librosa.amplitude_to_db(np.abs(np.fft.rfft(x)))
    
    h1bin = int(np.round(f0 * nfft / sr))
    h2bin = int(np.round(2 * f0 * nfft / sr))
    
    return x[h1bin] - x[h2bin]
    