# Implementation of a physical model of the vocal tract using the 
# Kelly-Lochbaum model.

# Translation of vocal tract used in https://dood.al/pinktrombone/, without nose tract

import numpy as np
from utils import frames_to_samples

class PhysicalVocalTract:

    def __init__(self, size=44) -> None:
        self.size = size

        self.blade_start = 10
        self.lip_start = 39
        
        self.diameter = np.zeros(self.size)
        self.rest_diameter = np.zeros(self.size)
        self.target_diameter = np.zeros(self.size)
        self.R = np.zeros(self.size)
        self.L = np.zeros(self.size)
        self.reflection = np.zeros(self.size + 1)
        self.new_reflection = np.zeros(self.size + 1)
        self.junction_outL = np.zeros(self.size + 1)
        self.junction_outR = np.zeros(self.size + 1)
        self.A = np.zeros(self.size)

        self.velum_target = 0.01
        self.glottal_reflection = 0.75
        self.lip_reflection = -0.85
    

    def process_input(self, input: np.array, diameters : np.array=None) -> np.array:
        n = input.size
        n_vt = self.size # number of pieces in the vocal tract

        # diameter shape: (frames, n_vt)
        A = diameters ** 2
        reflections = (A[:, :-1] - A[:, 1:]) / (A[:, :-1] + A[:, 1:])
        n_frames = diameters.shape[0]
        frame_size = input.size // ((n_frames - 1) if n_frames > 1 else n_frames)
        n = (n_frames - 1) * frame_size if n_frames > 1 else frame_size
        reflections = frames_to_samples(reflections, frame_size, numpy=True).reshape((n, n_vt-1))
        R = np.zeros(n_vt)
        L = np.zeros(n_vt)
        junction_outL = np.zeros(n_vt + 1)
        junction_outR = np.zeros(n_vt + 1)

        glottal_reflection = self.glottal_reflection
        lip_reflection = self.lip_reflection
        w = np.zeros(n_vt * 2)

        k = 0
        out = np.zeros(n*2+1)
        #out = np.zeros(n)

        while k < n * 2:
            junction_outR[0] = L[0] * glottal_reflection + input[k // 2]
            junction_outL[n_vt] = R[n_vt - 1] * lip_reflection

            w = reflections[k//2, :] * (R[:-1] + L[1:]) 
            junction_outR[1:-1] = R[:-1] - w
            junction_outL[1:-1] = L[1:] + w
            R[:n_vt] = junction_outR[:n_vt] * 0.999
            L[:n_vt] = junction_outL[1:n_vt+1] * 0.999

            out[k+1] += R[n_vt - 1]
            #out[k // 2] += R[n_vt - 1]

            k += 1

        out = out[1:] + out[:-1]
        return out[1::2] * 0.25
        #return out * 0.25