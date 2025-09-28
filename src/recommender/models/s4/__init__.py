"""From-scratch Structured State-Space (S4) implementation.

Public surface:
- HiPPO / DPLR initialization (:mod:`.hippo`)
- bilinear & ZOH discretization (:mod:`.discretization`)
- the DPLR kernel + FFT long-convolution (:mod:`.kernel`)
- trainable layer / residual block (:mod:`.s4_layer`)
"""

from .hippo import make_dplr_hippo, make_hippo, make_nplr_hippo

__all__ = ["make_hippo", "make_nplr_hippo", "make_dplr_hippo"]
