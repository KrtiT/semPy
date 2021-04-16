#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model example from semopy article."""
import pandas as pd
import numpy as np
import os

__desc = '''# Measurement part
eta1 =~ y1 + y2 + y3
eta2 =~ y3 + y2
eta3 =~ y4 + y5
eta4 =~ y4 + y6
# Structural part
eta3 ~ x2 + x1
eta4 ~ x3
x3 ~ eta1 + eta2 + x1
x4 ~ eta4 + x6
y7 ~ x4 + x6
# Additional covariances
y6 ~~ y5
x2 ~~ eta2'''


__folder = os.path.dirname(os.path.abspath(__file__))
__filename = '%s/article_data.csv' % __folder
__u_filename = '%s/article_data_u.npy' % __folder
__k_filename = '%s/article_data_k.npy' % __folder
__v_filename = '%s/article_data_u_vars.txt' % __folder
__params_filename = '%s/article_params.csv' % __folder

def get_model():
    """
    Retrieve model description in semopy syntax.

    Returns
    -------
    str
        Model's description.

    """
    return __desc


def get_data(drop_factors=True, random_effects=False):
    """
    Retrieve dataset.
    
    Parameters
    -------
    drop_factors : bool, optional
        If True, then factors are dropped from the dataframe. The default is
        True.
    random_effects : bool, optional
        If True, then data contaminated with random effects together with
        covariance matrix K is returned instead.

    Returns
    -------
    pd.DataFrame
        Dataset.

    """
    data = pd.read_csv(__filename, sep=',', index_col=0)
    if drop_factors:
        etas = [v for v in data.columns if v.startswith('eta')]
        data = data.drop(etas, axis=1)
    if random_effects:
        data['group'] = data.index
        k = np.load(__k_filename)
        k = pd.DataFrame(k, index=data['group'], columns=data['group'])
        u = np.load(__u_filename)
        with open(__v_filename, 'r') as f:
            cols = f.read().split(' ')
        data[cols] += u
        return data, k
    return data

def get_params():
    """
    Retrieve true parameter values.

    Returns
    -------
    None.

    """
    return pd.read_csv(__params_filename, sep=',', index_col=0)
    
