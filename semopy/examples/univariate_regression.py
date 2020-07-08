#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Univariate regression example model.

Correct parameter value is 5.0 (regression parameter).
"""
import pandas as pd

__desc = 'y ~ x'

__filename = '%s/univariate_data.csv' % '/'.join(__file__.split('/')[:-1])


def get_model():
    """
    Retrieve model description in semopy syntax.

    Returnsunivariate_regression
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
