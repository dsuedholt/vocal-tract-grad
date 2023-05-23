import torch

class ControlParameter():
    """
    Wrapper class to scale and constrain parameters to [0, 1] for easier
    gradient descent tuning
    """

    def __init__(self, shape, min_val=0, max_val=1, scale_fn=None, init_val=None, requires_grad=True):
        if scale_fn is None:
            # simple linear scaling        
            self.scale_fn = lambda x : (max_val - min_val) * x + min_val
        else:
            self.scale_fn = scale_fn
         
        self.data = torch.rand(shape, requires_grad=requires_grad)

        if init_val is not None:
            with torch.no_grad():
                self.data[:] = init_val
    
    def get_raw(self):
        return self.data
    
    def get_denormed(self):
        return self.scale_fn(self.data)
   
    def constrain(self):
        with torch.no_grad():
            self.data.clamp_(0, 1)