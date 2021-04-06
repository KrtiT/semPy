# -*- coding: utf-8 -*-
"""This is do-nothing effect that servers entirely for demonstrational and
testing purposes"""
from .effect_base import EffectBase
import numpy as np

class EffectBlank(EffectBase):
    """
    This effect does nothing. K matrix is just a static identity matrix.
    
    The only purposes of this effect are testing and providing guidelines for
    possible developers.
    """
    def __init__(self, d_mode='identity'):
        super().__init__(None, d_mode=d_mode)

    def load(self, order, model, data, clean_start=True, **kwargs):
        super().load(order, model, data, clean_start, **kwargs)
        self.mx_identity = np.identity(data.shape[0])
        if clean_start:
            self.parameters = np.array([])

    def calc_k(self, model):
        return self.mx_identity

    def calc_k_grad(self, model):
        return []

    