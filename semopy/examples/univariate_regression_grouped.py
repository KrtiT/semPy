#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Univariate regression with random effect (three groups) """
from typing import Tuple

import pandas as pd
import os

__folder = os.path.dirname(os.path.abspath(__file__))
__filename = f"{__folder}/univariate_regression_grouped_data.csv"
__filename_k = f"{__folder}/univariate_regression_grouped_kinship.csv"


def get_model() -> str:
    """
    Retrieve model description in Semopy syntax
    """
    return "y ~ x"


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retreive dataset and kinship matrix
    """
    data = pd.read_csv(__filename, index_col=0)
    k = pd.read_csv(__filename_k, index_col=0)
    k.columns = list(map(int, k.columns))
    return data, k
