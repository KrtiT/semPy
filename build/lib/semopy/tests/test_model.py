#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from ..model import Model
from ..examples import univariate_regression, multivariate_regression

np.random.seed(2021)
n = 100
p = 3
params = [np.random.uniform(0.2, 1.2, size=(p - 1, 1)),
          np.random.uniform(0.2, 1.2, size=(p -1, 1))]
params = list(map(lambda x: np.append([1], x), params))
y = np.random.normal(size=(n, 2 * p))
eta1 = np.random.normal(scale=1, size=(n, 1))
eta2 = np.random.normal(scale=1, size=(n, 1)) + 3 * eta1
y[:, :p] += np.kron(params[0], eta1)
y[:, p:] += np.kron(params[1], eta2)

res = list()
d = {'eta1': list(), 'eta2': list()}
y_names = list()
for i in range(1, p + 1):
    res.append((f'y{i}', '~', 'eta1', params[0][i - 1]))
    y_names.append(res[-1][0])
    d['eta1'].append(y_names[-1])
for j in range(1, p + 1):
    res.append((f'y{j + i}', '~', 'eta2', params[1][j - 1]))
    y_names.append(res[-1][0])
    d['eta2'].append(y_names[-1])
desc = '\n'.join(f"{eta} =~ {' + '.join(ys)}" for eta, ys in d.items())
desc += '\neta2 ~ eta1'
params = pd.DataFrame.from_records(res, columns=['lval', 'op', 'rval', 'est'])
data = pd.DataFrame(np.append(np.append(y, eta1, axis=1), eta2, axis=1),
                    columns=y_names + ['eta1', 'eta2'])


class TestModel(unittest.TestCase):
    def evaluate(self, desc: str, data: pd.DataFrame, true: pd.DataFrame,
                 obj='MLW'):
        m = Model(desc)
        r = m.fit(data, obj=obj)
        assert r.success, f"Optimization routine failed. [{obj}]"
        ins = m.inspect()
        errs = list()
        for _, row in true.iterrows():
            t = (ins['op'] == row['op']) & (ins['lval'] == row['lval']) &\
                (ins['rval'] == row['rval'])
            t = ins[t]
            try:
                assert t['p-value'].values[0] < 0.05,\
                       f"Incorrect p-value estimate [{obj}]."
            except TypeError:
                pass
            est = t['Estimate'].values[0]
            errs.append(abs((est - row['est']) / row['est']))
        err = np.mean(errs)
        assert err < 0.1, \
               f"Parameter estimation quality is too low: {err} [{obj}]"
    
    def test_univariate_regression(self):
        desc = univariate_regression.get_model()
        data = univariate_regression.get_data()
        true = pd.DataFrame([['y', '~', 'x', 5.0]],
                            columns=['lval', 'op', 'rval', 'est'])
        self.evaluate(desc, data, true, 'MLW')
        self.evaluate(desc, data, true, 'ULS')
        self.evaluate(desc, data, true, 'GLS')
        self.evaluate(desc, data - data.mean(), true, 'FIML')

    def test_multivariate_regression(self):
        desc = multivariate_regression.get_model()
        data = multivariate_regression.get_data()
        true = pd.DataFrame([['y', '~', 'x1', 2.0],
                             ['y', '~', 'x2', 6.0],
                             ['y', '~', 'x3', -10.0]],
                            columns=['lval', 'op', 'rval', 'est'])
        self.evaluate(desc, data, true, 'MLW')
        self.evaluate(desc, data, true, 'ULS')
        self.evaluate(desc, data, true, 'GLS')
        self.evaluate(desc, data - data.mean(), true, 'FIML')

    def test_random_model(self):
        global params
        global desc
        global data
        self.evaluate(desc, data, params, 'MLW')
        self.evaluate(desc, data, params, 'ULS')
        self.evaluate(desc, data, params, 'GLS')
        self.evaluate(desc, data - data.mean(), params, 'FIML')