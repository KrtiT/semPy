from semopy.examples import example_rf, univariate_regression_grouped, multivariate_regression, univariate_regression, example_article
from semopy import Model, ModelEffects, calc_stats
from semopy.stats import calc_dof, calc_chi2

import pandas as pd


def test_stats_redundant():
    """
    Test models with redundant single group
    """
    for case in (example_article, univariate_regression):
        desc, data = case.get_model(), case.get_data()
        data["group"] = 0
        k = pd.DataFrame({0: (1,)})
        model = ModelEffects(desc)
        model.fit(data, group="group", k=k)
        calc_stats(model=model, group="group", data=data, k=k)


def test_stats_univariate():
    desc, (data, k) = univariate_regression_grouped.get_model(), univariate_regression_grouped.get_data()
    model = ModelEffects(desc)
    model.fit(data, group="group", k=k)
    calc_stats(model, group="group", data=data, k=k)


def test_stats_rf():
    desc, (data, k) = example_rf.get_model(), example_rf.get_data()
    model = ModelEffects(desc)
    model.fit(data,group="group", k=k)
    calc_stats(model, group="group", data=data, k=k)
