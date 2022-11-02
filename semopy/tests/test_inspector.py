import semopy
from semopy import inspector
from semopy.examples import political_democracy as poldem

from pathlib import Path
import pandas as pd
import numpy as np


def test_rsquare():
    desc = poldem.get_model()
    data = poldem.get_data()
    mod = semopy.Model(desc)
    mod.fit(data)
    rsqare = inspector.calc_rsquare(mod)
    rsqare_reference = pd.read_csv(Path(__file__).parent.parent / "examples/pd_rsquare.csv",
                                   index_col=0)

    compare = rsqare_reference.join(rsqare, how="inner")
    abs_tol = .02
    assert (np.abs(compare.r2 - compare.r2Ref) < abs_tol).all()
