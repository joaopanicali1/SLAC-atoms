import torch 
import torch.nn as nn
import torch.nn.functional as F

def vibrational_positional_encoding(t_grid_fs, n_channels):
    """
    Positional encoding using frequencies spanning the vibrational range 
    rather than standard sinusoidal encoding. 
    """
    # frequency ranges in cm^-1
    freq_min = 5 
    freq_max = 4000

    # speed of lightin cm/fs
    c = 3e-5

    omega_min = 2 * torch.pi * c * freq_min
    omega_max = 2 * torch.pi * c * freq_max

    # log spaced frequencies for more resolution at low frequncies
    omegas = torch.logspace(
        torch.log10(torch.tensor(omega_min)),
        torch.log10(torch.tensor(omega_max)),
        n_channels // 2
    )

    t = t_grid_fs.unsqueeze(-1)
    phases = t * omegas.unsqueeze(0)
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)

