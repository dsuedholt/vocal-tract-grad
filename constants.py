import torch

NPOINTS = 1025
DOUBLE_PRECISION = True
SR = 44100

REAL_TYPE = torch.float64 if DOUBLE_PRECISION else torch.float32
COMPLEX_TYPE = torch.complex128 if DOUBLE_PRECISION else torch.complex64

Z = torch.exp(1j * torch.linspace(0, torch.pi, NPOINTS, requires_grad=False, dtype=REAL_TYPE))