import torch
from control_params import ControlParameter
from utils import frames_to_samples

# Glottal source using the LF-model, adapted from dood.al/pinktrombone
# to synthesize multiple frames at once / offline

class Glottis:
    def __init__(self, n_points=1025, sample_rate: float=48000):
        self.T = 1 / sample_rate
        self.n_samples = (n_points-1) * 2
        self.freq = 150
    
    def setup_lf(self, tenseness):
        Rd = 3 * (1 - tenseness)
        Rd = torch.clamp(Rd, 0.5, 2.7)

        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = 1 / (2 * Rg)
        Te = Tp + Tp * Rk

        epsilon = 1  / Ta
        shift = torch.exp(-epsilon * (1 - Te))
        delta = 1 - shift

        rhs_integral = (1 / epsilon) * (shift - 1) + (1 - Te) * shift
        rhs_integral /= delta

        lower_integral = -(Te - Tp)/2 + rhs_integral
        upper_integral = -lower_integral

        omega = torch.pi / Tp
        s = torch.sin(omega * Te)
        y = -torch.pi * s * upper_integral / (Tp * 2)
        z = torch.log(y)
        alpha = z / (Tp / 2  - Te)
        EO = -1 / (s * torch.exp(alpha * Te))

        self.alpha = alpha
        self.EO = EO
        self.epsilon = epsilon
        self.shift = shift
        self.delta = delta
        self.Te = Te
        self.omega = omega

    def set_frequency(self, freq):
        self.freq = freq
    
    def create_tenseness_param(self, init_val=None, requires_grad=True):
        return ControlParameter(
            shape=(1,),
            #min_val=1e-5,
            #max_val=1,
            scale_fn=lambda x: (x+1e-8)**2,
            init_val=init_val,
            requires_grad=requires_grad
        )
    
    def get_waveform(self, tenseness : torch.Tensor | ControlParameter, freq=None, frame_len=None):
        if type(tenseness) == ControlParameter:
            tenseness = tenseness.get_denormed()
    
        if frame_len is None:
            frame_len = self.n_samples
    
        if freq is None:
            freq = self.freq * torch.ones((1, 1))
        
        if len(tenseness.shape) == 1:
            tenseness = torch.unsqueeze(tenseness, dim=1)
        
        n_frames = tenseness.shape[0]
        if n_frames > 1: 
            n_frames -= 1
            n = frame_len * n_frames
            wav_len = 1 / torch.flatten(frames_to_samples(freq, frame_len))
            t = torch.arange(n) * self.T
            for i in range(n):
                if t[i] > wav_len[i]:
                    t[i:] -= wav_len[i]
            t /= wav_len
        
        else:
            n = frame_len
            wav_len = 1 / freq[0]
            t = torch.remainder(torch.arange(n) * self.T, wav_len) / wav_len

        result = torch.zeros(n)

        for i in range(n_frames):
            self.setup_lf(tenseness[i])
            idxs = torch.arange(i*frame_len, (i+1)*frame_len)

            t_slice = t[idxs]

            greaterIdx = t_slice > self.Te
            lesserIdx = t_slice <= self.Te

            result[idxs[greaterIdx]] = (-torch.exp(-self.epsilon * (t_slice[greaterIdx] - self.Te)) + self.shift) / self.delta
            result[idxs[lesserIdx]] = self.EO * torch.exp(self.alpha * t_slice[lesserIdx]) * torch.sin(self.omega * t_slice[lesserIdx])
            result[idxs] *= (tenseness[i]) ** 0.25

            aspiration = (1 - torch.sqrt(tenseness[i])) * 0.2 * (torch.rand(frame_len) - 0.5)
            aspiration *= 0.2

            result[idxs] += aspiration

        return result 