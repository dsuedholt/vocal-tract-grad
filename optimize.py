import torch
from tract_proxy import VocalTractProxy
from glottis import Glottis
from control_params import ControlParameter
from utils import log_mag_mse_loss, weighted_log_mag_mse_loss

NPOINTS = 1025


class OptimizerBase:
    def __init__(self):
        self.optims = []
        self.params = {}
        self.children = []
        self.state = {}
    
    def get_prediction(self):
        raise NotImplementedError()
    
    def get_regularizer(self):
        return self._get_regularizer() + sum([c.get_regularizer() for c in self.children])
        
    def _get_regularizer(self):
        return torch.zeros((1,))
    
    def zero_grad(self):
        for c in self.children:
            c.zero_grad()
        for o in self.optims:
            o.zero_grad()
    
    def step(self):
        for c in self.children:
            c.step()
        for o in self.optims:
            o.step()
        for p in self.params.values():
            p.constrain()

    def optimize(self, target, loss_fn=log_mag_mse_loss, n_iters=1000, log_every_n_iters=0):

        for n in range(n_iters):
            
            pred = self.get_prediction()
            # compare magnitude response of predicted control parameters to target
            base_loss = loss_fn(pred, target)
            loss = base_loss + self.get_regularizer()

            self.zero_grad()
            loss.backward()
            self.step()
            
            if log_every_n_iters > 0 and n % log_every_n_iters == 0:
                print(f'Iter: {n:4d}, loss: {base_loss.item():3.2f}')

    def get_controls(self):
        return {key: self.params[key].get_denormed().detach().clone() for key in self.params.keys()}


class TractControlsOptimizer(OptimizerBase):
    def __init__(self, vt : VocalTractProxy) -> None:
        super(TractControlsOptimizer, self).__init__()

        self.vt = vt

        p = {
            "tongue_idx": vt.create_tongue_idx_param(init_val=0.5),
            "tongue_diam": vt.create_tongue_diam_param(init_val=0.5),
            "constrictions": vt.create_constrictions_param(init_val=0.1)
        }
        
        self.params = p
        self.optims = [
            torch.optim.SGD([p["tongue_idx"].get_raw(), p["tongue_diam"].get_raw()], lr=1e-4, momentum=0.8),
            torch.optim.SGD([p["constrictions"].get_raw()], lr=1e-4, momentum=0.6)
        ]

    def get_prediction(self):
        p = self.params
        self.rest_diam = self.vt.apply_tongue(p["tongue_idx"], p["tongue_diam"])
        self.constricted_diam = self.vt.apply_constrictions(p["constrictions"], self.rest_diam)
        return self.vt.freq_response(self.constricted_diam)
        
    def _get_regularizer(self):
        # constrictions can occur at almost every segment
        # here, we apply three additional loss terms all meant
        # to penalize the area fit for relying on a lot of constrictions

        constrs_t = self.params["constrictions"].get_denormed()

        # total amount of constriction
        loss = 0.5 * torch.sum(torch.square(constrs_t)) 

        # second derivative of constriction function
        loss += torch.sum(torch.abs(constrs_t[2:] - 2 * constrs_t[1:-1] + constrs_t[:-2]))
        
        # penalize effect constrictions have on area function
        loss += 0.5 * torch.sum(torch.abs(self.rest_diam - self.constricted_diam))
        return loss


class DiameterOptimizer(OptimizerBase):
    def __init__(self, vt) -> None:
        super(DiameterOptimizer, self).__init__()
        self.vt = vt
        self.params = {
            "diameter": vt.create_diameter_param()
        }
        self.optims = [
            torch.optim.SGD([self.params["diameter"].get_raw()], lr=1e-3, momentum=0.99)
        ]

    def get_prediction(self):
        return self.vt.freq_response(self.params["diameter"])
    

class WaveformGlottisOptimizer(OptimizerBase):
    def __init__(self, glottis):
        super(WaveformGlottisOptimizer, self).__init__()
        self.glottis = glottis
        self.params = {
            "tenseness": glottis.create_tenseness_param(0.8)
        }

        self.optims = [
            torch.optim.SGD([self.params["tenseness"].get_raw()], lr=1e-4)
        ]
    
    def get_prediction(self):
        return self.glottis.get_waveform(self.params["tenseness"])
        

class EndToEndOptimizer(OptimizerBase):
    def __init__(self, glottis_optim, tract_optim):
        super(EndToEndOptimizer, self).__init__()
        self.glottis_optim = glottis_optim
        self.tract_optim = tract_optim
        self.children = [glottis_optim, tract_optim]
    
    def get_prediction(self):
        g = self.glottis_optim.get_prediction()
        t = self.tract_optim.get_prediction()
        return g * t



if __name__ == "__main__":
    vt = VocalTractProxy(NPOINTS)
    target_diam = torch.tensor([0.6,0.6,0.6,0.6,0.6,0.6,0.6,1.1,1.1,1.1,0.4570080204629816,0.3476969359723392,0.33333333333333326,0.3476969359723392,0.39043406432232075,0.4604923884469041,0.5561468398958948,0.6750420886156946,0.8142505389921147,0.9703444169705286,1.1394801732292281,1.3174931241197307,1.5000000000000002,1.6825068758802693,1.860519826770772,2.029655583029472,2.185749461007885,2.3249579113843057,2.443853160104106,2.539507611553096,2.6095659356776792,2.6523030640276604,2.666666666666667,2.6523030640276604,2.6095659356776792,2.5395076115530957,2.443853160104105,2.275460436701247,2.0485995688063077,1.5,1.5,1.5,1.5,1.5])
    target_response = vt.freq_response(target_diam)

    vt_optim = TractControlsOptimizer(vt)

    glott = Glottis(n_points=NPOINTS)

    vt_optim.optimize(target_response, loss_fn=weighted_log_mag_mse_loss)

