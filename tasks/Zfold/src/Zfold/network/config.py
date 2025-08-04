import numpy as np
import torch

bins_size = 1
dist_max = 40
dist_min = 3
atoms = {
    "C3'": {'cutoff': 13},
    "P": {'cutoff': 13},
    "N1": {'cutoff': 13},
    # "C5'": {'cutoff': 13},
    # "O4'": {'cutoff': 13},
    "C4": {'cutoff': 13},
    # "O5'": {'cutoff': 13},
    # "C4'": {'cutoff': 13},
    "C1'": {'cutoff': 13},
}

obj = {
    '2D': {'distance': ["C3'", "P", "N1", "C4", "C1'",'CiNj','PiNj'],
           'contact': ['all atom'],
           },
}

n_bins = {
    '2D': {
        'distance': 38,
        'angle': 13,
        'dihedral_angle': 25,
        'contact': 1
    }
}
