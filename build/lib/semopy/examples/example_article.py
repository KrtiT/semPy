#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model example from semopy article."""
import pandas as pd

__desc = '''# structural part
eta3 ~ x1 + x2
eta4 ~ x3
x3 ~ eta1 + eta2 + x1 + x4
x4 ~ eta4
x5 ~ x4
# measurement part
eta1 =~ y1 + y2 + y3
eta2 =~ y3
eta3 =~ y4 + y5
eta4 =~ y4 + y6
# additional covariances
eta2 ~~   x2
y5 ~~   y6'''

__filename = '%s/article_data.csv' % '/'.join(__file__.split('/')[:-1])


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
    return pd.read_csv(__filename, sep=',', index_col=0)
