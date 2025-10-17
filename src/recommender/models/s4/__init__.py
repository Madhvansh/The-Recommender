"""From-scratch Structured State-Space (S4) implementation.

Public surface:
- HiPPO / DPLR initialization (:mod:`.hippo`)
- bilinear & ZOH discretization (:mod:`.discretization`)
- the DPLR kernel + FFT long-convolution (:mod:`.kernel`)
- trainable layer / residual block (:mod:`.s4_layer`)
"""

from .discretization import discretize, discretize_bilinear, discretize_zoh
from .hippo import make_dplr_hippo, make_hippo, make_nplr_hippo
from .kernel import fft_conv, s4_kernel_dplr, s4_recurrent_step
from .s4_layer import S4Block, S4Layer

__all__ = [
    "make_hippo",
    "make_nplr_hippo",
    "make_dplr_hippo",
    "discretize",
    "discretize_bilinear",
    "discretize_zoh",
    "s4_kernel_dplr",
    "fft_conv",
    "s4_recurrent_step",
    "S4Layer",
    "S4Block",
]
