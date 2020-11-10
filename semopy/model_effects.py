# -*- coding: utf-8 -*-
"""Random Effects SEM."""
import pandas as pd
import numpy as np
from .model_means import ModelMeans
from .utils import chol_inv, chol_inv2, cov, kron_identity, calc_zkz
from scipy.linalg import block_diag
from .solver import Solver
import logging


class ModelEffects(ModelMeans):
    """
    Random Effects model.

    Random Effects SEM can be interpreted as a generalization of Linear Mixed
    Models (LMM) to SEM.
    """

    matrices_names = tuple(list(ModelMeans.matrices_names) + ['d'])
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
            default is False.

        Returns
        -------
        None.

        """
        self.dict_effects[self.symb_rf_covariance] = self.effect_rf_covariance
        super().__init__(description, mimic_lavaan=mimic_lavaan,
                         baseline=baseline, intercepts=intercepts)
        self.objectives = {'REML': (self.obj_reml, self.grad_reml),
                           'REML2': (self.obj_reml2, self.grad_reml2),
                           'ML': (self.obj_matnorm, self.grad_matnorm)}

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
        Exceptio
            Rises when group parameter is None.
        Returns
        -------
        None.

        """
        if data is None:
            if not hasattr(self, 'mx_data'):
                raise Exception("Data must be provided.")
            return
        else:
            data = data.copy()
        if group is None:
            raise Exception('Group name (column) must be provided.')
        obs = self.vars['observed']
        exo = self.vars['observed_exogenous']
        if self.intercepts:
            data = data.copy()
            data['1'] = 1.0
        cols = data.columns
        missing = (set(obs) | set(exo)) - set(set(cols))
        if missing:
            t = ', '.join(missing)
            raise KeyError('Variables {} are missing from data.'.format(t))
        self.load_data(data, k=k, covariance=cov, group=group)
        self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()

    def _fit(self, obj='REML', solver='SLSQP'):
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

    def fit(self, data=None, group=None, k=None, cov=None, obj='ML',
            solver='SLSQP', clean_slate=False, regularization=None):
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
            Objective function to minimize. Possible values are 'REML', 'ML'.
            The default is 'REML'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        clean_slate : bool, optional
            If False, successive fits will be performed with previous results
            as starting values. If True, parameter vector is reset each time
            prior to optimization. The default is False.
        regularization
            Special structure as returend by create_regularization function.
            If not None, then a regularization will be applied to a certain
            parameters in the model. The default is None.

        Raises
        ------
        Exception
            Rises when attempting to use MatNorm in absence of full data.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        self.load(data=data, cov=cov, group=group, k=k,
                  clean_slate=clean_slate)
        if not hasattr(self, 'mx_data'):
            raise Exception('Full data must be supplied.')
        if obj == 'REML':
            if self.__loaded != 'REML':
                self.load_reml()
            self.calc_fim = self.calc_fim_reml
            res_reml = self._fit(obj='REML', solver=solver)
            self.load_ml(fake=True)
            sigma, (self.mx_m, _) = self.calc_sigma()
            self.mx_r_inv = chol_inv(self.calc_r(sigma))
            self.mx_w_inv = self.calc_w_inv(sigma)[0]
            res_reml2 = self._fit(obj='REML2', solver=solver)
            return (res_reml, res_reml2)
        elif obj == 'ML':
            if self.__loaded != 'ML':
                self.load_ml()
            res = self._fit(obj='ML', solver=solver)
            return res
        else:
            raise NotImplementedError(f'Unknown objective {obj}.')

    def predict(self, data: pd.DataFrame, group: str, k: pd.DataFrame,
                ret_opt=False):
        raise NotImplementedError('ModelEffects can''t predict right now.')
        from .imputer import ImputerEffects
        imp = ImputerEffects(self, data, group, k)
        res = imp.fit(solver='SLSQP')
        data = imp.get_fancy()
        return data if not ret_opt else (data, res)

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
        mx = self.mx_d
        rows, cols = self.names_d
        for lv, rvs in items.items():
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

    def set_fim_means(self):
        """
        Substitute true FIM matrix with means-only FIM matrix.

        A trick to reduce GWAS time as only mean components are subject to
        analysis.
        Returns
        -------
        None.

        """
        
        self.calc_fim = self.calc_fim_means

    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ------------------for covariance structure computations-------------------
    '''

    '''
    ---------------------------R and W matrices-------------------------------
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
        n = self.num_n
        return n * sigma + self.mx_d * self.trace_zkz

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
        n = self.num_n
        for g, df in zip(sigma_grad, self.mx_diffs):
            g = n * g
            if df[6] is not None:  # D
                g += df[6] * self.trace_zkz
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
        w = self.calc_w(sigma)
        if np.any(w < 1e-9):
            raise np.linalg.LinAlgError
        return 1 / w, np.sum(np.log(w))

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
        return tr_d * self.mx_s + tr_sigma

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
                g = np.trace(g) * self.mx_i_n
            if df[6] is not None:  # D
                g += np.trace(df[6]) * self.mx_s
            grad.append(g)
        return grad

    def calc_w_inv_grad(self, inv_w: np.ndarray, sigma_grad: list):
        """
        Calculate gradient of W inverse and logdet matrix.

        Parameters
        ----------
        inv_w : np.ndarray
            Inverse of W matrix.
        sigma_grad : list
            Gradient of Sigma matrix.

        Returns
        -------
        grad : list
            Gradient of inverse of W.
        grad_logdet : list
            Gradient of logdet of W.

        """
        grad, grad_logdet = list(), list()
        inv_w_t = inv_w * self.mx_s
        inv_w_d = -(inv_w ** 2)
        inv_w_d_t = inv_w_d * self.mx_s
        for g, df in zip(sigma_grad, self.mx_diffs):
            gw = g
            gl = g
            if len(g.shape):
                tr = np.trace(g)
                gw = tr * inv_w_d
                gl = tr * inv_w
            if df[6] is not None:  # D
                tr = np.trace(df[6])
                gw += tr * inv_w_d_t
                gl += tr * inv_w_t
            grad.append(gw)
            grad_logdet.append(np.sum(gl))
        return grad, grad_logdet

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
        k : pd.DataFrame or tuple
            Covariance matrix betwen groups. If None, then it's assumed to be
            an identity matrix. Alternatively, a tuple of (ZKZ^T, S, Q) can be
            provided where ZKZ^T = Q S Q^T an eigendecomposition of ZKZ^T. S
            must be provided in the vector/list form. The default is None.
        covariance : pd.DataFrame, optional 
            Custom covariance matrix. The default is None.

        Returns
        -------
        None.

        """
        obs = self.vars['observed']
        if type(k) in (tuple, list):
            if len(k) != 3:
                raise Exception("Both ZKZ^T and its eigendecomposition must "
                                "be provided.")
        self.mx_g_orig = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g_orig.shape) != 2:
            self.mx_g_orig = self.mx_g_orig[np.newaxis, :]
        self.mx_g = self.mx_g_orig
        self.mx_data = data[obs].values
        self.num_m = len(set(self.vars['observed']) - self.vars['latent'])
        if type(k) is tuple:
            self.mx_zkz, self.mx_sk, self.mx_q = k
            self._ktuple = True
        else:
            self._ktuple = False
            self.mx_zkz = calc_zkz(data[group], k)
        self.__loaded = None
        self.load_cov(covariance[obs].loc[obs]
                      if covariance is not None else cov(self.mx_data))

    def load_ml(self, fake=False):
        self.trace_zkz = np.trace(self.mx_zkz)
        if self._ktuple:
            self.mx_s = self.mx_sk
            q = self.mx_q
        else:
            s, q = np.linalg.eigh(self.mx_zkz)
            self.mx_s = s
        self.mx_data_transformed = self.mx_data.T @ q
        self.mx_g = self.mx_g_orig @ q
        self.num_n = self.mx_data_transformed.shape[1]
        self.mx_i_n = np.ones(self.num_n)
        if not fake:
            self.calc_fim = self.calc_fim_ml
        self.__loaded = 'ML'

    def load_reml(self):
        g = self.mx_g_orig
        try:
            s = np.identity(g.shape[1]) - g.T @ chol_inv(g @ g.T) @ g
        except ValueError:
            raise Exception("REML should not be used when there are no"
                            " either intercepts or exogenous variables in "
                            "Gamma matrices.")
        d, q = np.linalg.eigh(s)
        rank_dec = 0
        for i in d:
            if abs(i) < 1e-8:
                rank_dec += 1
            else:
                break
        d = np.diag(d)[rank_dec:, :]
        a = d @ q.T
        azkza = a @ self.mx_zkz @ a.T
        self.trace_zkz = np.trace(azkza)
        s, q = np.linalg.eigh(azkza)
        self.mx_s = s
        self.mx_data_transformed = self.mx_data.T @ a.T @ q
        self.num_n = self.mx_data_transformed.shape[1]
        self.mx_i_n = np.ones(self.num_n)
        self.calc_fim = self.calc_fim_reml
        self.__loaded = 'REML'

    '''
    ---------------Matrix Variate Normal Restricted Maximum Likelihood---------
    '''

    def obj_reml(self, x: np.ndarray):
        """
        Restricted loglikelihood of matrix-variate normal distribution.

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
        sigma, _ = self.calc_sigma()
        try:
            r = self.calc_r(sigma)
            r_inv, logdet_r = chol_inv2(r)
            w_inv, logdet_w = self.calc_w_inv(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        mx = self.mx_data_transformed
        tr_r = np.trace(r)
        n, m = self.num_n, self.num_m
        r_center = r_inv @ mx
        center_w = mx * w_inv
        tr = tr_r * np.einsum('ji,ji->', center_w, r_center)
        return tr + m * logdet_w + n * logdet_r - n * m * np.log(tr_r)

    def grad_reml(self, x: np.ndarray):
        """
        Gradient of REML objective of matrix-variate normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of REML objective.

        """
        self.update_matrices(x)
        grad = np.zeros_like(x)
        sigma, (m, c) = self.calc_sigma()
        try:
            r = self.calc_r(sigma)
            r_inv = chol_inv(r)
            w_inv, _ = self.calc_w_inv(sigma)
        except np.linalg.LinAlgError:
            grad[:] = np.inf
            return grad
        center = self.mx_data_transformed
        A = r_inv @ center
        B = (center * w_inv).T
        tr_ab = np.einsum('ij,ji->', A, B)
        tr_r = np.trace(r)
        V1 = center.T @ A
        V3 = B @ r_inv
        V2 = self.num_n * r_inv / tr_r - A @ V3
        sigma_grad = self.calc_sigma_grad(m, c)
        r_grad = self.calc_r_grad(sigma_grad)
        w_grad, w_grad_logdet = self.calc_w_inv_grad(w_inv, sigma_grad)
        n, m = self.num_n, self.num_m
        for i, (d_r, d_w, d_l) in enumerate(zip(r_grad, w_grad,
                                                w_grad_logdet)):
            g = 0.0
            tr_long = 0.0
            if len(d_r.shape):
                tr_long += np.einsum('ij,ji->', V2, d_r)
                tr_dr = np.trace(d_r)
                g += tr_dr * tr_ab
                g -= m * n * tr_dr / tr_r
            if len(d_w.shape):
                tr_long += np.einsum('ii,i->', V1, d_w)
                g += m * d_l
            g += tr_r * tr_long
            grad[i] = g
        return grad

    '''
    ------------------Matrix Variate REML (II-nd stage)-----------------------
    '''

    def obj_reml2(self, x: np.ndarray):
        """
        Loglikelihood of matrix-variate normal distribution given Sigma.

        For a second stage of REML estimation.
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
        mean = self.calc_mean(self.mx_m)
        center = self.mx_data_transformed - mean
        r_center = self.mx_r_inv @ center
        center_w = center * self.mx_w_inv
        return np.einsum('ji,ji->', center_w, r_center)

    def grad_reml2(self, x: np.ndarray):
        """
        Gradient of loglikelihood of matrix-variate normal distribution.

        For a second stage of REML estimation.
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
        center = self.mx_data_transformed - self.calc_mean(self.mx_m)
        t =  (center * self.mx_w_inv).T @ self.mx_r_inv
        mean_grad = self.calc_mean_grad_reml()
        for i, g in enumerate(mean_grad):
            if len(g.shape):
                grad[i] = -2 * np.einsum('ij,ji->', g, t)
        return grad

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
            return np.inf
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        tr_r = np.trace(r)
        m, n = self.num_m, self.num_n
        r_center = r_inv @ center
        center_w = center * w_inv
        tr = tr_r * np.einsum('ij,ij->', center_w, r_center)
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
            grad[:] = np.inf
            return grad
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        A = r_inv @ center
        B = (center * w_inv).T
        tr_ab = np.einsum('ij,ji->', A, B)
        tr_r = np.trace(r)
        V1 = center.T @ A
        V3 = B @ r_inv
        V2 = self.num_n * r_inv / tr_r - A @ V3
        
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        r_grad = self.calc_r_grad(sigma_grad)
        w_grad, w_grad_logdet = self.calc_w_inv_grad(w_inv, sigma_grad)
        n, m = self.num_n, self.num_m
        for i, (d_m, d_r, d_w, d_l) in enumerate(zip(mean_grad, r_grad,
                                                     w_grad, w_grad_logdet)):
            g = 0.0
            tr_long = 0.0
            if len(d_m.shape):
                tr_long -= 2 * np.einsum('ij,ji->', V3, d_m)
            if len(d_r.shape):
                tr_long += np.einsum('ij,ji->', V2, d_r)
                tr_dr = np.trace(d_r)
                g += tr_dr * tr_ab
                g -= m * n * tr_dr / tr_r
            if len(d_w.shape):
                tr_long += np.einsum('ii,i->', V1, d_w)
                g += m * d_l
            g += tr_r * tr_long
            grad[i] = g
        return grad

    '''
    -----------------------Fisher Information Matrix---------------------------
    '''

    def calc_fim_reml(self, inverse=False):
        """
        Calculate Fisher Information Matrix when estimation was performed via
        REML.

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
        sigma, aux = self.calc_sigma()
        w_inv = self.calc_w_inv(sigma)[0]
        r = self.calc_r_reml2(sigma)
        r_inv = chol_inv(r)
        m, c = aux
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        w_grad = self.calc_w_grad(sigma_grad)
        r_grad = self.calc_r_grad(sigma_grad)
        sigma = np.kron(w_inv, r_inv)
        n = self.mx_data_transformed.shape[1]
        m = self.num_m
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
        inds_base = list()
        sgs = list()
        for i, (g_wr, g_mean, pm) in enumerate(zip(wr, mean_grad, prod_means)):
            if g_wr is not None or g_mean is not None:
                sgs.append((g_wr, g_mean, pm))
                inds_base.append(i)

        w_inv = np.diag(self.calc_w_inv_reml()[0].flatten())
        r_inv, _, tr_r = self.calc_r_inv_reml()
        m, c = aux
        w_grad = self.calc_w_reml_grad()
        r_grad = self.calc_r_reml_grad()
        sigma = np.kron(w_inv, r_inv)
        n = self.reml_mx_data_transformed.shape[1]
        m = self.num_m
        i_im = np.identity(n * m) / tr_r
        wr = [kron_identity(w_inv @ dw, m) + kron_identity(r_inv @ dr, n, True)
              if len(dw.shape) else None for dw, dr in zip(w_grad, r_grad)]
        wr = [wr - i_im * np.trace(dr) if wr is not None else None
              for wr, dr in zip(wr, r_grad)]
        rfs = list()
        inds_rf = list()
        for i, g in enumerate(wr):
            if g is not None:
                rfs.append(g)
                inds_rf.append(i)
        sz = len(inds_base)
        mx_base = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(i, sz):
                if sgs[i][0] is not None and sgs[j][0] is not None:
                    mx_base[i, j] = np.einsum('ij,ji->', sgs[i][0],
                                              sgs[j][0]) / 2
                elif sgs[i][1] is not None and sgs[j][2] is not None:
                    mx_base[i, j] += np.einsum('ij,ji->', sgs[i][1], sgs[j][2])
        mx_base = mx_base + np.triu(mx_base, 1).T
        sz = len(inds_rf)
        mx_rf = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(i, sz):
                mx_rf[i, j] = np.einsum('ij,ij->', rfs[i], rfs[j])
        mx_rf = mx_rf + np.triu(mx_rf, 1).T
        inds_base = np.array(inds_base, dtype=np.int)
        inds_rf = np.array(inds_rf, dtype=np.int)
        inds = np.append(inds_base, inds_rf)
        fim = block_diag(mx_base, mx_rf)
        fim = fim[:, inds][:, inds]
        if inverse:
            try:
                mx_base_inv = chol_inv(mx_base)
                mx_rf_inv = chol_inv(mx_rf)
                self._fim_warn = False
            except np.linalg.LinAlgError:
                logging.warn("Fisher Information Matrix is not PD."
                             "Moore-Penrose inverse will be used instead of "
                             "Cholesky decomposition. See "
                             "10.1109/TSP.2012.2208105.")
                self._fim_warn = True
                mx_base_inv = np.linalg.pinv(mx_base)
                mx_rf_inv = np.linalg.pinv(mx_rf)
            fim_inv = block_diag(mx_base_inv, mx_rf_inv)
            fim_inv = fim_inv[inds, :][:, inds]
            return (fim, fim_inv)
        return fim

    def calc_fim_ml(self, inverse=False):
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
        sigma = np.kron(np.diag(w_inv), r_inv)
        sz = len(sigma_grad)
        n, m = self.num_n, self.num_m
        tr_r = np.trace(r)
        i_im = np.identity(n * m) / tr_r
        wr = [kron_identity(np.diag(w_inv * dw), m) \
              + kron_identity(r_inv @ dr, n, True)
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
                    info[i, k] += prod_means[i] @ mean_grad[k]
        fim = info + np.triu(info, 1).T
        fim = fim
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim

    def calc_fim_means(self, inverse=False):
        """
        Calculate Fisher Information Matrix for mean components only.

        Exponential-family distributions are assumed. Useful to fascilate GWAS
        as we usually don't care about other parameters.
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
        mean_grad = self.calc_mean_grad(m, c)
        w_inv = self.calc_w_inv(sigma)[0]
        r = self.calc_r(sigma)
        r_inv = chol_inv(r)
        sigma = np.kron(np.diag(w_inv), r_inv)
        sz = len(self.param_vals)
        m = self.num_n, self.num_m
        tr_r = np.trace(r)
        mean_grad = [g.reshape((-1, 1), order="F") if len(g.shape) else None
                     for g in mean_grad]
        prod_means = [g.T @ sigma * tr_r if g is not None else None
                      for g in mean_grad]
        info = np.zeros((sz, sz))
        for i in range(sz):
            for k in range(i, sz):
                if prod_means[i] is not None and mean_grad[k] is not None:
                    info[i, k] += prod_means[i] @ mean_grad[k]
        fim = info + np.triu(info, 1).T
        fim = fim
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim
