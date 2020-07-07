# -*- coding: utf-8 -*-
"""Random Effects SEM."""
import pandas as pd
import numpy as np
from .model_means import ModelMeans
from .utils import chol_inv, chol_inv2, cov, kron_identity
from .solver import Solver
import logging


class ModelEffects(ModelMeans):
    """
    Random Effects model.

    Random Effects SEM can be interpreted as a generalization of Linear Mixed
    Models (LMM) to SEM.
    """

    matrices_names = tuple(list(ModelMeans.matrices_names) + ['d', 'v'])
    symb_rf_covariance = '~R~'

    def __init__(self, description: str, mimic_lavaan=False, baseline=False,
                 intercepts=True):
        """
        Instantiate Random Effects SEM.

        Parameters
        ----------
        description : str
            Model description in semopy syntax.

        mimic_lavaan: bool
            If True, output variables are correlated and not conceptually
            identical to indicators. lavaan treats them that way, but it's
            less computationally effective. The default is False.

        baseline : bool
            If True, the model will be set to baseline model.
            Baseline model here is an independence model where all variables
            are considered to be independent with zero covariance. Only
            variances are estimated. The default is False.

        intercepts: bool
            If True, intercepts are also modeled. Intercept terms can be
            accessed via "1" symbol in a regression equation, i.e. x1 ~ 1. The
            default is True.

        Returns
        -------
        None.

        """
        self.dict_effects[self.symb_rf_covariance] = self.effect_rf_covariance
        super().__init__(description, mimic_lavaan=mimic_lavaan,
                         baseline=baseline, intercepts=intercepts)
        self.objectives = {'MatNorm': (self.obj_matnorm, self.grad_matnorm)}

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply random effect variance
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        super().preprocess_effects(effects)
        for v in self.vars['observed']:
            if v not in self.vars['latent']:  # Workaround for Imputer
                t = effects[self.symb_rf_covariance][v]
                if v not in t:
                    t[v] = None
        t = effects[self.symb_rf_covariance]['1']
        if '1' not in t:
            t['1'] = None

    def build_d(self):
        """
        D matrix is a covariance matrix for random effects across columns.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['observed']
        n = len(names)
        mx = np.zeros((n, n))
        return mx, (names, names)

    def build_v(self):
        """
        Build "v" value -- a variance parameter.

        v is a variance parameter that is used as a multiplicator for identity
        matrix.
        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = ['1']
        n = len(names)
        mx = np.zeros((n, n))
        return mx, (names, names)

    def load(self, data, group: str, k=None, cov=None, clean_slate=False):
        """
        Load dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns as variables.
        group : str
            Name of column with group labels.
        k : pd.DataFrame
            Covariance matrix across rows, i.e. kinship matrix. If None,
            identity is assumed. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. Used only for variance
            starting values. The default is None.
        clean_slate : bool, optional
            If True, resets parameters vector. The default is False.

        KeyError
            Rises when there are missing variables from the data.

        Returns
        -------
        None.

        """
        if data is None:
            if not hasattr(self, 'mx_data'):
                raise Exception("Data must be provided.")
            return
        obs = self.vars['observed']
        exo = self.vars['observed_exogenous']
        if self.intercepts:
            data = data.copy()
            data['1'] = 1.0
        cols = data.columns
        missing = (set(obs) | exo) - set(set(cols))
        if missing:
            t = ', '.join(missing)
            raise KeyError('Variables {} are missing from data.'.format(t))
        self.load_data(data, k=k, covariance=cov, group=group)
        self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()

    def fit(self, data=None, group=None, k=None, cov=None, obj='MatNorm',
            solver='SLSQP', groups=None, clean_slate=False):
        """
        Fit model to data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        group : str
            Name of column in data with group labels. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        obj : str, optional
            Objective function to minimize. Possible values are 'MLW', 'FIML',
            'ULS', 'GLS'. The default is 'MLW'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        clean_slate : bool, optional
            If False, successive fits will be performed with previous results
            as starting values. If True, parameter vector is reset each time
            prior to optimization. The default is False.

        Raises
        ------
        Exception
            Rises when attempting to use MatNorm in absence of full data.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        self.load(data=data, cov=cov, group=group,
                  clean_slate=clean_slate)
        if obj == 'MatNorm':
            if not hasattr(self, 'mx_data'):
                raise Exception('Full data must be supplied for FIML')
            self.prepare_fiml()
        fun, grad = self.get_objective(obj)
        solver = Solver(solver, fun, grad, self.param_vals,
                        constrs=self.constraints,
                        bounds=self.get_bounds())
        res = solver.solve()
        res.name_obj = obj
        self.param_vals = res.x
        self.update_matrices(res.x)
        self.last_result = res
        return res

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError('Prediction feature for ModelEffects is \
                                  temporarily disabled.')

    def effect_rf_covariance(self, items: dict):
        """
        Work through random effects covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        for lv, rvs in items.items():
            if lv == '1':
                mx = self.mx_v
                rows, cols = self.names_v
            else:
                mx = self.mx_d
                rows, cols = self.names_d
            for rv, mult in rvs.items():
                name = None
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    active = True
                    if mult is not None:
                        if mult != self.symb_starting_values:
                            name = mult
                        else:
                            active = False
                    val = None
                if name is None:
                    self.n_param_cov += 1
                    name = '_c%s' % self.n_param_cov
                i, j = rows.index(lv), cols.index(rv)
                ind = (i, j)
                if i == j:
                    bound = (0, None)
                    symm = False
                else:
                    if self.baseline:
                        continue
                    bound = (None, None)
                    symm = True
                self.add_param(name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=symm, bound=bound)

    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ------------------for covariance structure computations-------------------
    '''

    def calc_r(self, sigma: np.ndarray):
        """
        Calculate covariance across columns matrix R.

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        tuple
            R matrix.

        """
        n = self.mx_data.shape[0]
        s = n * self.mx_v[0, 0]
        return n * sigma + self.mx_d * self.trace_aka + s

    def calc_r_grad(self, sigma_grad: list):
        """
        Calculate gradient of R matrix.

        Parameters
        ----------
        sigma_grad : list
            Sigma gradient values.

        Returns
        -------
        grad : list
            Gradient of R matrix.

        """
        grad = list()
        n = self.mx_data.shape[0]
        for g, df in zip(sigma_grad, self.mx_diffs):
            g = n * g
            if df[6] is not None:  # D
                g += df[6] * self.trace_aka
            if df[7] is not None:  # v
                g += self.mx_ones_m
            grad.append(g)
        return grad

    def calc_w_inv(self, sigma: np.ndarray):
        """
        Calculate inverse and logdet of covariance across rows matrix W.

        This function estimates only inverse of W. There was no need in package
        to estimate W.
        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        tuple
        R^{-1} and ln|R|.

        """
        tr_sigma = np.trace(sigma)
        tr_d = np.trace(self.mx_d)
        f_inv = (tr_d * self.mx_s + self.num_m * self.mx_v[0, 0])
        if np.any(f_inv < 1e-10):
            raise np.linalg.LinAlgError
        logdet_f = np.sum(np.log(f_inv))
        f_inv = f_inv ** (-1)
        # Sherman-Morrison:
        a = f_inv.T * self.mx_qtt_square * f_inv
        form = tr_sigma * np.sum(self.mx_oq_square * f_inv) + 1
        if form < 1e-10:
            raise np.linalg.LinAlgError
        f_inv = np.diag(f_inv.flatten())
        w_inv = f_inv - tr_sigma * a / form
        # Determinant Lemma
        logdet_w = logdet_f + np.log(form)
        return w_inv, logdet_w

    def calc_w(self, sigma: np.ndarray):
        """
        Calculate W matrix for testing purposes.

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix.

        Returns
        -------
        np.ndarray
            W matrix.

        """
        tr_sigma = np.trace(sigma)
        tr_d = np.trace(self.mx_d)
        f = (tr_d * self.mx_s + self.num_m * self.mx_v[0, 0])
        return self.mx_qtt_square * tr_sigma + np.diag(f.flatten()) 

    def calc_w_grad(self, sigma_grad: list):
        """
        Calculate gradient of W matrix.

        Parameters
        ----------
        sigma_grad : list
            Gradient of Sigma matrix.

        Returns
        -------
        grad : list
            Gradient of W.

        """
        grad = list()
        for g, df in zip(sigma_grad, self.mx_diffs):
            if len(g.shape):
                g = np.trace(g) * self.mx_qtt_square
            if df[6] is not None:  # D
                g += np.trace(df[6]) * self.mx_s_diag
            if df[7] is not None:  # v
                g += self.mx_i_n
            grad.append(g)
        return grad

    '''
    ---------------------Preparing structures for a more-----------------------
    ------------------------efficient computations-----------------------------
    '''

    def load_data(self, data: pd.DataFrame, group: str, k=None,
                  covariance=None):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        group : str
            Name of column that correspond to group labels.
        K : pd.DataFrame
            Covariance matrix betwen groups. If None, then it's assumed to be
            an identity matrix.
        covariance : pd.DataFrame, optional
            Custom covariance matrix. The default is None.

        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        grs = data[group]
        p_names = list(grs.unique())
        p, n = len(p_names), data.shape[0]
        if k is None:
            k = np.identity(p)
        elif k.shape[0] != p:
            raise Exception("Dimensions of K don't match number of groups.")
        z = np.zeros((n, p))
        for i, germ in enumerate(grs):
            j = p_names.index(germ)
            z[i, j] = 1.0
        if type(k) is pd.DataFrame:
            k = k.loc[p_names, p_names].values
        aka = z @ k @ z.T
        self.trace_aka = np.trace(aka)
        s, q = np.linalg.eigh(aka)
        self.mx_s = s[np.newaxis, :]
        self.mx_s_diag = np.diag(s)
        oq = np.sum(q, axis=0, keepdims=True)
        self.mx_qtt_square = np.outer(oq, oq)
        self.mx_oq_square = oq.flatten() ** 2

        self.mx_data = data[obs].values
        self.mx_data_transformed = self.mx_data.T @ q
        self.mx_g1 = data[self.vars['observed_exogenous_1']].values.T @ q
        self.mx_g2 = data[self.vars['observed_exogenous_2']].values.T @ q
        self.mx_q = q
        self.num_m = len(set(self.vars['observed']) -self.vars['latent'])
        self.mx_i_n = self.num_m * np.identity(n)
        self.mx_ones_m = n * np.ones((self.num_m, self.num_m))
        self.load_cov(covariance[obs].loc[obs]
                      if covariance is not None else cov(self.mx_data))

    '''
    ------------------Matrix Variate Normal Maximum Likelihood-----------------
    '''

    def obj_matnorm(self, x: np.ndarray):
        """
        Loglikelihood of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Loglikelihood.

        """
        self.update_matrices(x)
        sigma, (m, _) = self.calc_sigma()
        try:
            r = self.calc_r(sigma)
            r_inv, logdet_r = chol_inv2(r)
            w_inv, logdet_w = self.calc_w_inv(sigma)
        except np.linalg.LinAlgError:
            return np.nan
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        tr_r = np.trace(r)
        n, m = self.mx_data.shape
        r_center = r_inv @ center
        center_w = center @ w_inv
        tr = tr_r * np.einsum('ji,ji->', center_w, r_center)
        return tr + m * logdet_w + n * logdet_r - n * m * np.log(tr_r)

    def grad_matnorm(self, x: np.ndarray):
        """
        Gradient of loglikelihood of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of MatNorm objective.

        """
        self.update_matrices(x)
        grad = np.zeros_like(x)
        sigma, (m, c) = self.calc_sigma()
        try:
            r = self.calc_r(sigma)
            r_inv = chol_inv(r)
            w_inv, _ = self.calc_w_inv(sigma)
        except np.linalg.LinAlgError:
            grad[:] = np.nan
            return grad
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        center_t = center.T
        wm = w_inv @ center_t
        wcr = wm @ r_inv
        rm = r_inv @ center
        a = wcr @ wm.T
        b = rm @ wcr
        tr_l = np.einsum('ij,ji->', wcr, center)
        tr_r = np.trace(r)
        wcr2 = 2 * wcr
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        r_grad = self.calc_r_grad(sigma_grad)
        w_grad = self.calc_w_grad(sigma_grad)
        n, m = self.mx_data.shape
        for i, (d_m, d_r, d_w) in enumerate(zip(mean_grad, r_grad, w_grad)):
            g = 0.0
            tr_long = 0.0
            if len(d_m.shape):
                tr_long += np.einsum('ij,ji->', wcr2, d_m)
            if len(d_r.shape):
                tr_long += np.einsum('ij,ji->', d_r, b)
                g += n * np.einsum('ij,ji->', r_inv, d_r)
                tr_dr = np.trace(d_r)
                g += tr_l * tr_dr
                g -= m * n * tr_dr / tr_r
            if len(d_w.shape):
                tr_long += np.einsum('ij,ji->', d_w, a)
                g += m * np.einsum('ij,ji->', w_inv, d_w)
            g -= tr_r * tr_long
            grad[i] = g
        return grad

    '''
    -----------------------Fisher Information Matrix---------------------------
    '''

    def calc_fim(self, inverse=False):
        """
        Calculate Fisher Information Matrix.

        Exponential-family distributions are assumed.
        Parameters
        ----------
        inverse : bool, optional
            If True, function also returns inverse of FIM. The default is
            False.

        Returns
        -------
        np.ndarray
            FIM.
        np.ndarray, optional
            FIM^{-1}.

        """
        sigma, (m, c) = self.calc_sigma()
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        w_inv = self.calc_w_inv(sigma)[0]
        r = self.calc_r(sigma)
        r_inv = chol_inv(r)
        w_grad = self.calc_w_grad(sigma_grad)
        r_grad = self.calc_r_grad(sigma_grad)
        sigma = np.kron(w_inv, r_inv)
        sz = len(sigma_grad)
        n, m = self.mx_data.shape
        tr_r = np.trace(r)
        i_im = np.identity(n * m) / tr_r
        wr = [kron_identity(w_inv @ dw, m) + kron_identity(r_inv @ dr, n, True)
              if len(dw.shape) else None for dw, dr in zip(w_grad, r_grad)]
        wr = [wr - i_im * np.trace(dr) if wr is not None else None
              for wr, dr in zip(wr, r_grad)]                
        mean_grad = [g.reshape((-1, 1), order="F") if len(g.shape) else None
                     for g in mean_grad]
        prod_means = [g.T @ sigma * tr_r if g is not None else None
                      for g in mean_grad]
        info = np.zeros((sz, sz))
        for i in range(sz):
            for k in range(i, sz):
                if wr[i] is not None and wr[k] is not None:
                    info[i, k] = np.einsum('ij,ji->', wr[i], wr[k]) / 2
                if prod_means[i] is not None and mean_grad[k] is not None:
                    info[i, k] +=  prod_means[i] @ mean_grad[k]
        fim = info + np.triu(info, 1).T
        fim = 2 * fim
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim
