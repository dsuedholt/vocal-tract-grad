import argparse
import librosa
import torch
from utils import decompose, weighted_log_mag_mse_loss, frames_to_samples, yin, freqz, h1h2
from optimize import WaveformGlottisOptimizer, TractControlsOptimizer
from tract_proxy import VocalTractProxy
from physical_tract import PhysicalVocalTract
from glottis import Glottis
from tqdm import tqdm
from matplotlib import pyplot as plt
import soundfile as sf
import numpy as np
import time

FMIN, FMAX = 70, 500

class Analyzer:
    def __init__(self, glottis, tract, sr):
        self.sr = sr
        self.glottis = glottis
        self.n_points = tract.n_points
        self.win = torch.hamming_window((self.n_points-1)*2, periodic=True)

        self.tract_optim = TractControlsOptimizer(tract)

    def analyze_frame(self, frame, n_iters=100, true_resp=None, true_source=None, return_if_resp=False):
        f0 = yin(frame, sr=self.sr)
        
        # result of inverse filtering
        if_glott, if_coeffs = decompose(frame)

        if true_source is not None:
            if_glott = true_source

        # frequency response to match
        if true_resp is not None:
            if_resp = true_resp
        else:
            if_resp = torch.tensor(freqz(if_coeffs, n_points=self.n_points), dtype=torch.complex64)

        self.glottis.set_frequency(f0)

        Rd = torch.tensor((h1h2(if_glott, f0, sr=self.sr) + 7.6) / 11.1)
        
        tenseness = torch.clamp(1 - Rd / 3, 0, 1)
        
        self.tract_optim.optimize(if_resp, loss_fn=weighted_log_mag_mse_loss, n_iters=n_iters)

        if return_if_resp:
            return f0, tenseness, self.tract_optim.get_controls(), if_resp
        else:
            return f0, tenseness, self.tract_optim.get_controls()

def main(args):
    audio, sr = librosa.load(args.input_file, sr=args.sample_rate)

    fl = args.frame_length
    hl = args.hop_length

    frames = librosa.util.frame(audio, frame_length=fl, hop_length=hl, axis=0)
    n_frames = frames.shape[0]
    
    n_points = fl // 2 + 1

    glottis = Glottis(n_points, sr)
    tract = VocalTractProxy(n_points)

    analyzer = Analyzer(glottis, tract, sr)

    f0s = torch.zeros((n_frames, 1))
    tenses = torch.zeros((n_frames, 1))
    diameters = torch.zeros((n_frames, tract.base_diam.numel()))
    rest_diameters = torch.zeros((n_frames, tract.base_diam.numel()))

    start = time.time()
    print("Predicting parameters...")
    for i in tqdm(range(n_frames)):
        f0, tenseness, tract_controls = analyzer.analyze_frame(frames[i, :])
        f0s[i, :] = f0
        tenses[i, :] = tenseness 
        diameters[i, :] = tract.apply_tongue(tract_controls['tongue_idx'], tract_controls['tongue_diam'])
        rest_diameters[i, :] = tract.apply_constrictions(tract_controls['constrictions'], diameters[i, :])
    stop = time.time()    
    
    print("Done. Real-time factor (vs length of input):", (stop - start) / (n_frames * hl / sr))
    
    print("Resynthesizing...")

    tenses = torch.concat([tenses, torch.ones((1, 1)) * tenses[-1]], dim=0)
    tenses = frames_to_samples(tenses, args.upsample_glottis)
    tenses = torch.reshape(tenses, (tenses.shape[0], 1))

    f0s = torch.concat([f0s, torch.ones((1, 1)) * f0s[-1]], dim=0)
    f0s = frames_to_samples(f0s, args.upsample_glottis)
    f0s = torch.reshape(f0s, (f0s.shape[0], 1))
    
    #tenses = torch.clamp_max(tenses * args.tenseness_factor, 1)
    out = glottis.get_waveform(tenses, f0s, hl // args.upsample_glottis).numpy()
    pvt = PhysicalVocalTract()
    out = pvt.process_input(out, diameters.detach().numpy())
    sf.write(args.output_file, out, sr)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-fl', '--frame_length', type=int, default=4096)
    parser.add_argument('-hl', '--hop_length', type=int, default=4096)
    parser.add_argument('-ut', '--upsample_glottis', type=int, default=8)
    parser.add_argument('-sr', '--sample_rate', type=int, default=44100)
    main(parser.parse_args())