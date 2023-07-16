import torch
import math
from utils import make_z
from control_params import ControlParameter

class VocalTractProxy:
    def __init__(self, n_points) -> None:

        self.n_points = n_points

        self.tongue_diam_min = 2.05
        self.tongue_diam_max = 3.5

        self.tongue_idx_min = 12
        self.tongue_idx_max = 29
        self.tongue_idx_center = (self.tongue_idx_max + self.tongue_idx_min) / 2

        self.constr_idx_min = 2
        self.constr_idx_max = 43

        self.constr_diam_min = 0.1
        self.constr_diam_max = 3
        
        self.blade_start = 10
        self.lip_start = 39
        self.tip_start = 32
        self.grid_offset = 1.7
        
        self.loss_factor = 0.999

        self.constr_idxs = torch.arange(self.constr_idx_min, self.constr_idx_max+1) * 1.0
        self.full_idxs = torch.unsqueeze(self.constr_idxs, dim=1).repeat_interleave(self.constr_idxs.numel(), dim=1) 
        
        self.relpos = torch.abs(self.full_idxs - self.constr_idxs)
        self.relpos = torch.clamp_min(self.relpos, 0)
        
        self.width = 10-5*(self.full_idxs.T-25)/(self.tip_start-25)
        self.width = torch.clamp(self.width, 5, 10)
        self.width_mask = (self.relpos < self.width) * 1.0

        self.I = torch.eye(2, requires_grad=False) + 0j
        self.I = self.I.reshape((1, 2, 2)).repeat_interleave(n_points, 0)
        
        self.z = make_z(n_points)

        self.r0 = 0.75 # R_0, reflection at the glottis
        self.rl = -0.85 # R_L, reflection at the lips
        self.tl = 1 + self.rl
        
        self.tongue_idxs = torch.arange(self.blade_start, self.lip_start) * 1.0
        self.curve_mod = torch.ones_like(self.tongue_idxs)
        self.curve_mod[0] = 0.94
        self.curve_mod[-1] = 0.8
        self.curve_mod[-2] = 0.94
        
        self.base_diam = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.1, 1.1, 1.1, 1.1, 1.1, 1.5,
       1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
       1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
       1.5, 1.5, 1.5, 1.5, 1.5])
    
    
    def create_tongue_idx_param(self, init_val=None, requires_grad=True):
        return ControlParameter(
            shape=(1,),
            min_val=self.tongue_idx_min,
            max_val=self.tongue_idx_max,
            init_val=init_val,
            requires_grad=requires_grad
        )
    
    def create_tongue_diam_param(self, init_val=None, requires_grad=True):
        return ControlParameter(
            shape=(1,),
            min_val=self.tongue_diam_min,
            max_val=self.tongue_diam_max,
            init_val=init_val,
            requires_grad=requires_grad
        )
    
    def create_constrictions_param(self, init_val=None, requires_grad=True):
        return ControlParameter(
            shape=(self.base_diam.numel() - self.constr_idx_min,),
            init_val=init_val,
            requires_grad=requires_grad
        )

    def create_diameter_param(self, init_val=None, requires_grad=True):
        return ControlParameter(
            shape=self.base_diam.shape,
            min_val=self.constr_diam_min,
            max_val=self.constr_diam_max,
            init_val=init_val,
            requires_grad=requires_grad
        )

    def freq_response(self, diam : torch.Tensor | ControlParameter, delay=1.0):
        """
        calculate magnitude response from diameters by evaluating the 
        transfer function at all values in self.z
        
        for details on transfer function see https://zenodo.org/record/5045182
        """
        
        if type(diam) == ControlParameter:
            diam = diam.get_denormed()
        
        # area
        A = diam ** 2
        
        # reflection coefficients 
        ks = (A[:-1] - A[1:]) / (A[:-1] + A[1:]) + 0j

        # Smyth assumes lossless, but PT implementation has loss at each junction
        z1 = self.loss_factor * self.z**(-1)

        K = self.I

        for i in range(len(ks)):
            mul = torch.ones(len(z1), 2, 2) + 0j

            mul[:, 1, 0] = ks[i]
            mul[:, 0, 1] = ks[i] * z1
            mul[:, 1, 1] = z1

            K = torch.matmul(K, mul)
        
        numer = self.tl * self.z**(-(len(ks) + 1)/2) * torch.prod(1 + ks)
        denom = K[:, 0, 0] + K[:, 0, 1] * self.rl - self.r0 * (K[:, 1, 0] + K[:, 1, 1] * self.rl) * self.z**(-1)

        # vocal tract operates at double samplerate and adds "intermediate" samples
        # this corresponds to additional 1 + z^(-1) lowpass filter in the transfer function
        lpf = 1 + self.z**(-delay)
        
        return lpf * numer / denom


    def apply_tongue(self, tongue_idx : torch.Tensor | ControlParameter, tongue_diam : torch.Tensor | ControlParameter, curr_diam=None):
        if type(tongue_idx) == ControlParameter:
            tongue_idx = tongue_idx.get_denormed()
        
        if type(tongue_diam) == ControlParameter:
            tongue_diam = tongue_diam.get_denormed()

        if curr_diam is None:
            curr_diam = self.base_diam

        t = 1.1 * torch.pi * (tongue_idx - self.tongue_idxs) / (self.tip_start - self.blade_start)
        fixedTongueDiameter = 2 + (tongue_diam - 2) / 1.5 
        curve = (1.5 - fixedTongueDiameter + self.grid_offset) * torch.cos(t) * self.curve_mod
        new_diam = 1.5 - curve
        return torch.cat([curr_diam[:self.blade_start], new_diam, curr_diam[self.lip_start:]])
    
        
    def apply_constrictions(self, constrs : torch.Tensor | ControlParameter, curr_diam):
        if type(constrs) == ControlParameter:
            constrs = constrs.get_denormed()
        
        #return torch.cat([curr_diam[:self.constr_idx_min], curr_diam[self.constr_idx_min:] * (1 - constrs) + self.constr_diam_min * constrs])
        return torch.cat([curr_diam[:self.constr_idx_min], curr_diam[self.constr_idx_min:] * (1 - constrs)])
    
    def apply_single_constriction(self, constr_idx : float, constr_diam, curr_diam):
        
        if type(constr_diam) == ControlParameter:
            constr_diam = constr_diam.get_denormed()


        curr_diam = torch.tensor(curr_diam)        

        width = 2
        if constr_idx < 25: width = 10
        elif constr_idx > self.tip_start: width = 5
        else: width = 10 - 5*(constr_idx-25)/(self.tip_start-25)
        
        int_idx = round(constr_idx)
        
        i = -math.ceil(width) - 1
        while i < width+1:
            if (int_idx + i) >= 0 and (int_idx + i) < len(curr_diam):
                relpos = int_idx + i - constr_idx
                relpos = abs(relpos) - 0.5
                if relpos <= 0: shrink = 0
                elif relpos > width: shrink = 1
                else: shrink = 0.5*(1 - math.cos(math.pi * relpos / width))
                if curr_diam[int_idx + i] > constr_diam:
                    curr_diam[int_idx + i] = constr_diam + (curr_diam[int_idx + i] - constr_diam) * shrink
            i += 1
        
        return curr_diam