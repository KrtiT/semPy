# -*- coding: utf-8 -*-
"""Second order de-biasing correction"""
from .model_generation import generate_data
from copy import deepcopy

def bias_correction(model, n=100):
    model_c = deepcopy(model)
    t = None
    try:
        n_samples = model.n_samples
    except AttributeError:
        n_samples = model.mx_data[0]
    for _ in range(n):
        r = model_c.fit(generate_data(model, n=n_samples), clean_slate=True)
        if t is None:
            t = r.x
        else:
            t += r.x
    model.param_vals = 2 * model.param_vals - t / n
    
