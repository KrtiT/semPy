import semopy
import numpy as np
import pandas as pd

np.random.seed(123)

N = 100
eta1 = np.random.normal(size=N)
eta2 = np.random.normal(size=N)
eta1 += 0.3 * eta2

y1 = np.random.normal(size=N, scale=0.5) + eta1
y2 = np.random.normal(size=N, scale=0.5) + 2 * eta1
y3 = np.random.normal(size=N, scale=0.5) + 3 * eta1 + eta2
y4 = np.random.normal(size=N, scale=0.5) - eta2
y5 = np.random.normal(size=N, scale=0.5) + 1.5 * eta2
x = np.random.normal(size=N)
y3_card = np.zeros(y3.shape)
y3_card[y3 > 0] = 1
# y3_card[y3 > 0.4] = 2
data = pd.DataFrame([y1, y2, y3_card, y4, y5, x],
                    index=['y1', 'y2', 'y3', 'y4', 'y5', 'x']).T
data_tr = pd.DataFrame([y1, y2, y3, y4, y5, x],
                       index=['y1', 'y2', 'y3', 'y4', 'y5', 'x']).T
# c_est = semopy.polycorr.hetcor(data_tr, ['y3'])
c_tr = data_tr.corr()
c_f  = data.corr()
desc = '''eta1 =~ y1 + y2 + y3
eta2 =~ y3 + y4 + y5
eta1 ~ eta2
'''
# m = semopy.Model(desc)
# # data -= data.mean()
# data /= data.std()
# res = m.fit(data_tr)
# print(m.inspect())

desc = semopy.efa.explore_cfa_model(data_tr )
# desc = '''eta1 =~ y1 + y2 + y3 + y5
# eta2 =~ y5 + y4 + y3'''
print(desc)
m=semopy.Model(desc)
m.fit(data_tr)
m.inspect()

data, model = semopy.examples.political_democracy.get_data(), semopy.examples.political_democracy.get_model()
m = semopy.Model(model)
m.fit(data)
print(m.inspect(std_est=True))