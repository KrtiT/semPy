#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An example of dataset with random effects.

Correct parameter estimates:
       lval  op  rval     Value
        0    x2   ~  eta1  0.492118
        1  eta2   ~    x1  2.553225
        2  eta2   ~    x2 -1.722126
        3  eta1  =~    y2 -0.952966
        4  eta2  =~    y4 -1.921717
Notice that get_data() returns tuple: dataset and K matrix.
"""
import pandas as pd

__desc = '''eta1 =~ y1 + y2
eta2 =~ y3 + y4
x2 ~ eta1
eta2 ~ x1 + x2'''

__filename = '%s/example_rf_data.csv' % '/'.join(__file__.split('/')[:-1])
__filename_k = '%s/example_rf_kinship.csv' % '/'.join(__file__.split('/')[:-1])


def get_model():
    """
    Retrieve model description in semopy syntax.

    Returns
    -------
    str
        Model's description.

    """
    return __desc


def get_data():
    """
    Retrieve dataset and kinship matrix.

    Returns
    -------
    pd.DataFrame
        Dataset and K matrix.

    """
    data = pd.read_csv(__filename, index_col=0)
    data['group'] = data.index
    return data, pd.read_csv(__filename_k, index_col=0)
