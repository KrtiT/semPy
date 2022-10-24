from semopy.examples import example_rf, univariate_regression_grouped, multivariate_regression, univariate_regression
from semopy import Model, ModelEffects, calc_stats
from semopy.stats import calc_dof, calc_chi2


def test_stats_univariate():
    desc, (data, k) = univariate_regression_grouped.get_model(), univariate_regression_grouped.get_data()
    model = ModelEffects(desc)
    model.fit(data, group="group", k=k)
    calc_stats(model)


def test_stats_rf():
    desc, (data, k) = example_rf.get_model(), example_rf.get_data()
    model = ModelEffects(desc)
    model.fit(data,
              group="group", k=k
              )
    calc_stats(model)

    # dof = calc_dof(model)
    # calc_chi2(model, dof)
