#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Political Democracy model and data.

Bollen's Data on Industrialization and Political Democracy.
Dataset set includes four measures of democracy at two points in time, 1960
and 1965, and three measures of industrialization in 1960, for 75 developing
countries.
"""
import pandas as pd

__desc = '''# measurement model
ind60 =~ x1 + x2 + x3
dem60 =~ y1 + y2 + y3 + y4
dem65 =~ y5 + y6 + y7 + y8
# regressions
dem60 ~ ind60
dem65 ~ ind60 + dem60
# residual correlations
y1 ~~ y5
y2 ~~ y4 + y6
y3 ~~ y7
y4 ~~ y8
y6 ~~ y8'''

__filename = '%s/pd_data.csv' % '/'.join(__file__.split('/')[:-1])


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
    Retrieve dataset.

    Returns
    -------
    pd.DataFrame
        Dataset.

    """
    return pd.read_csv(__filename, index_col=0)
