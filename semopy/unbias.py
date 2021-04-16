# -*- coding: utf-8 -*-
"""Second order de-biasing correction"""
from .model_generation import generate_data
from copy import deepcopy
import pandas as pd

def bias_correction(model, n=100, resample_mean=False, extra_data=None,
                    **kwargs):
    model_c = deepcopy(model)
    t = None
    try:
        n_samples = model.n_samples
    except AttributeError:
        n_samples = model.mx_data[0]
    if extra_data is not None:
        extra_cols = set(extra_data.columns)
    for _ in range(n):
        if not resample_mean:
            data = generate_data(model, n=n_samples)
        else:
            data = generate_data(model, n=n_samples, generator_exo=None)
        if extra_data is not None:
            data.index = extra_data.index
            st = extra_cols - set(data.columns)
            data = pd.concat([data, extra_data[st]], axis=1)
        r = model_c.fit(data, clean_slate=True, **kwargs)
        if t is None:
            t = r.x
        else:
            t += r.x
    model.param_vals = 2 * model.param_vals - t / n
    
