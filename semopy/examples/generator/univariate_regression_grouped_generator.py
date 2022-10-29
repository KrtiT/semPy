"""
Generate Linear Mixed Model regression example
"""

import numpy as np
import pandas as pd
import seaborn as sns


def generate():
    n_groups = 3
    samples_per_group = 15
    # y = a + b*x + eps + (ups1|group), eps ~ sigma, ups1 ~ lambda1
    a, b, sigma, lambda1 = 1.5, .75, .5, 5

    np.random.seed(0)  # for reproducibility
    n_samples = n_groups * samples_per_group
    x = np.random.uniform(-3, 3, n_samples)
    eps = np.random.normal(0, sigma, n_samples)
    groups = np.array(range(n_groups)).repeat(samples_per_group)
    ups1 = np.random.normal(0, lambda1, n_groups).repeat(samples_per_group)
    y = a + b * x + eps + ups1
    data_random_bias = pd.DataFrame({
        "x": x,
        "y": y,
        "group": groups
    })
    plot_random_bias = sns.scatterplot(x="x", y="y", data=data_random_bias, hue="group")
    plot_random_bias.get_figure().savefig("lmm_random_bias.png")
    data_random_bias.to_csv("../univariate_regression_grouped_data.csv")

    # TODO produce meaningful kinship matrix
    k_proto = np.random.uniform(0, 1, (n_groups, n_groups))
    k = k_proto @ k_proto.T
    pd.DataFrame(k, columns=list(range(n_groups))).to_csv("../univariate_regression_grouped_kinship.csv")


if __name__ == "__main__":
    generate()
