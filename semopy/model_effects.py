# -*- coding: utf-8 -*-
"""Random Effects SEM."""
import pandas as pd
import numpy as np
from .model_means import ModelMeans
from .utils import chol_inv, chol_inv2, cov, kron_identity
from scipy.linalg import block_diag
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

    def fit(self, data=None, group=None, k=None, cov=None, obj='REML',
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
        if obj == 'REML' and self.reml_tr_f < 1e-9:
            logging.warning('ML instead of REML will be used as random '\
                            'effects variance components might be too '\
                            'unstable to estimate. Naturally, this issue '\
                            'arises in case of group size equal to the size '\
                            'of dataset. Then, no concern should be present '\
                            'as this is a normal behaviour.')
            obj = 'ML'
        if obj == 'REML':
            self.calc_fim = self.calc_fim_reml
            res_reml = self._fit(obj='REML', solver=solver)
            n = self.mx_data_transformed.shape[1]
            tr_d = np.trace(self.mx_d)
            f = tr_d * self.mx_s + self.num_m * self.mx_v[0, 0]
            self.reml_mx_f_inv = 1 / f
            self.reml_mx_f_p = np.diag(f.flatten())
            self.reml_mx_f_p_inv = np.diag(self.reml_mx_f_inv.flatten())
            self.reml_logdet_f = np.sum(np.log(f))
            self.reml_mx_ak = self.reml_mx_f_inv.T * self.mx_qtt_square * \
                self.reml_mx_f_inv
            self.reml_oqsum = np.sum(self.mx_oq_square * self.reml_mx_f_inv)
            s = n * self.mx_v[0, 0]
            self.reml_mx_d_r = self.mx_d * self.trace_zkz + s
            res_reml2 = self._fit(obj='REML2', solver=solver)
            return (res_reml, res_reml2)
        elif obj == 'ML':
            self.calc_fim = self.calc_fim_ml
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

    def effect_covariance(self, items: dict):
        """
        Work through covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        inners = self.vars['inner']
        for lv, rvs in items.items():
            lv_is_inner = lv in inners
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
                rv_is_inner = rv in inners
                if name is None:
                    self.n_param_cov += 1
                    name = '_c%s' % self.n_param_cov
                if lv_is_inner and rv_is_inner:
                    mx = self.mx_psi
                    rows, cols = self.names_psi
                else:
                    # continue
                    mx = self.mx_theta
                    rows, cols = self.names_theta
                    if lv_is_inner != rv_is_inner:
                        logging.info('Covariances between _outputs and \
                                     inner variables are not recommended.')
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

    '''
    --------------------R and W matrices for full ML---------------------------
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
        return n * sigma + self.mx_d * self.trace_zkz + s

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
                g += df[6] * self.trace_zkz
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
    --------------------R and W matrices for REML------------------------------
    '''

    def calc_r_reml(self):
        d = self.mx_d * self.reml_tr_f
        return self.reml_mx_ones_m * self.reml_tr_s * self.mx_v[0, 0] + d

    def calc_r_inv_reml(self):
        d = self.mx_d.diagonal() * self.reml_tr_f
        if np.any(d < 1e-9):
            raise np.linalg.LinAlgError
        t = self.reml_tr_s * self.mx_v[0, 0]
        m = len(d)
        tr_r = np.sum(d) + t * m
        logdet_d = np.sum(np.log(d))
        d = 1 / d
        tr_d = np.sum(d)
        td = (d)[:, None]
        denom = 1 + t * tr_d
        if abs(denom) < 1e-9:
            raise np.linalg.LinAlgError
        logdet = logdet_d + np.log(denom)
        return (np.diag(d) - td @ td.T *(t / denom)), logdet, tr_r

    def calc_r_reml_grad(self):
        """
        Calculate gradient of R matrix.

        Returns
        -------
        grad : list
            Gradient of R matrix.

        """
        grad = list()
        for df in self.mx_diffs:
            g = np.float32(0.0)
            if df[6] is not None:  # D
                g += df[6] * self.reml_tr_f
            if df[7] is not None:  # v
                g += self.reml_mx_ones_m * self.reml_tr_s
            grad.append(g)
        return grad

    def calc_w_reml(self):
        m = self.reml_mx_data_transformed.shape[0]
        return m * self.mx_v[0, 0] + np.trace(self.mx_d) * self.reml_mx_f

    def calc_w_inv_reml(self):
        w = self.calc_w_reml()
        if np.any(w < 1e-9):
            raise np.linalg.LinAlgError
        return 1 / w, np.sum(np.log(w))

    def calc_w_inv_reml_grad(self):
        m = self.reml_mx_data_transformed.shape[0]
        w = self.calc_w_reml()
        w_inv = 1 / w
        wt = -(w_inv ** 2)
        wd = wt * self.reml_mx_f
        wv = wt * m
        wd_logdet = -np.sum(wd * w)
        wv_logdet = -np.sum(wv * w)
        grad = list()
        grad_logdet = list()
        for df in self.mx_diffs:
            if df[6] is not None:  # D
                tr = np.trace(df[6])
                g = wd * tr
                g1 = wd_logdet * tr
            elif df[7] is not None:  # v
                g = wv
                g1 = wv_logdet
            else:
                g = np.float32(0.0)
                g1 = np.float32(0.0)
            grad.append(g)
            grad_logdet.append(g1)
        return grad, grad_logdet

    def calc_w_reml_grad(self):
        """
        Calculate gradient of W matrix.

        Returns
        -------
        grad : list
            Gradient of W.

        """
        grad = list()
        for df in self.mx_diffs:
            g = np.float32(0.0)
            if df[6] is not None:  # D
                g += np.trace(df[6]) * self.reml_mx_f_diag
            if df[7] is not None:  # v
                g += self.reml_mx_i_n
            grad.append(g)
        return grad

    '''
    --------------------R and W matrices for REML (II)-------------------------
    ---------------------------Second stage------------------------------------
    '''

    def calc_r_reml2(self, sigma: np.ndarray):
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
        n = self.mx_data_transformed.shape[1]
        return n * sigma + self.reml_mx_d_r

    def calc_r_reml2_grad(self, sigma_grad: list):
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
        for g in sigma_grad:
            grad.append(n * g)
        return grad

    def calc_w_reml2_inv(self, sigma: np.ndarray):
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
        # tr_d = np.trace(self.mx_d)
        # f_inv = (tr_d * self.mx_s + self.num_m * self.mx_v[0, 0])
        # if np.any(f_inv < 1e-10):
        #     raise np.linalg.LinAlgError
        # logdet_f = np.sum(np.log(f_inv))
        # f_inv = f_inv ** (-1)
        f_inv = self.reml_mx_f_inv
        # Sherman-Morrison:
        a = f_inv.T * self.mx_qtt_square * f_inv
        a = self.reml_mx_ak

        form = tr_sigma * np.sum(self.mx_oq_square * f_inv) + 1
        form = tr_sigma * self.reml_oqsum + 1
        if form < 1e-10:
            raise np.linalg.LinAlgError
        f_inv = self.reml_mx_f_p_inv
        w_inv = f_inv - tr_sigma * a / form
        # Determinant Lemma
        logdet_w = self.reml_logdet_f + np.log(form)
        return w_inv, logdet_w

    def calc_w_reml2(self, sigma: np.ndarray):
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
        return self.mx_qtt_square * tr_sigma + self.reml_mx_f_p

    
    def calc_w_reml2_grad(self, sigma_grad: list):
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
        for g in sigma_grad:
            if len(g.shape):
                g = np.trace(g) * self.mx_qtt_square
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
            try:
                k = k.loc[p_names, p_names].values
            except KeyError:
                raise KeyError("Certain groups in K differ from those "\
                                "provided in a dataset.")
        self.mx_g = data[self.vars['observed_exogenous']].values.T
        if len(self.mx_g.shape) != 2:
            self.mx_g = self.mx_g[np.newaxis, :]
        g = self.mx_g
        self.mx_data = data[obs].values
        self.num_m = len(set(self.vars['observed']) - self.vars['latent'])
        # REML-related structures
        s = np.identity(g.shape[1]) - g.T @ chol_inv(g @ g.T) @ g
        self.reml_tr_s = np.trace(s)
        d, q = np.linalg.eigh(s)
        rank_dec = 0
        for i in d:
            if abs(i) < 1e-8:
                rank_dec += 1
            else:
                break
        d = np.diag(d)[rank_dec:, :]
        self.reml_mx_a = d @ q.T
        zkz = z @ k @ z.T
        self.mx_z = z
        azkza = self.reml_mx_a @ zkz @ self.reml_mx_a.T
        self.reml_trace_azkza = np.trace(azkza)
        f, q = np.linalg.eigh(azkza)
        self.reml_mx_f = f[np.newaxis, :]
        self.reml_mx_f_diag = np.diag(f)
        self.reml_tr_f = np.sum(f)
        self.reml_mx_data_transformed = self.mx_data.T @ self.reml_mx_a.T @ q
        n = self.reml_mx_data_transformed.shape[1]
        self.reml_mx_ones_m = np.ones((self.num_m, self.num_m))
        self.reml_mx_i_n = self.num_m * np.identity(n)

        # ML-related structures
        self.mx_zkz = zkz
        self.trace_zkz = np.trace(zkz)
        s, q = np.linalg.eigh(zkz)
        self.mx_q = q
        self.mx_s = s[np.newaxis, :]
        self.mx_s_diag = np.diag(s)
        oq = np.sum(q, axis=0, keepdims=True)
        self.mx_qtt_square = np.outer(oq, oq)
        self.mx_oq_square = oq.flatten() ** 2
        self.mx_data_transformed = self.mx_data.T @ q
        self.mx_g = self.mx_g @ q
        n = self.mx_data_transformed.shape[1]
        # self.mx_g = self.mx_g @ q
        # self.mx_q = q
        self.mx_i_n = self.num_m * np.identity(n)
        self.mx_ones_m = n * np.ones((self.num_m, self.num_m))
        self.load_cov(covariance[obs].loc[obs]
                      if covariance is not None else cov(self.mx_data))

    '''
    ---------------Matrix Variate Normal Restricted Maximum Likelihood---------
    '''

    def obj_reml(self, x: np.ndarray):
        self.update_matrices(x)
        n = self.reml_mx_data_transformed.shape[1]
        m = self.num_m
        try:
            r_inv, r_logdet, tr_r = self.calc_r_inv_reml()
            w_inv, w_logdet = self.calc_w_inv_reml()
        except np.linalg.LinAlgError:
            return np.nan
        mx = self.reml_mx_data_transformed
        wmx = w_inv.T * mx.T
        rmx = r_inv @ mx
        tr = tr_r * np.einsum('ij,ji->', wmx, rmx)
        log = n * r_logdet + m * w_logdet - n * m * np.log(tr_r)
        return tr + log

    def grad_reml(self, x: np.ndarray):
        """
        Gradient of restricted ML of matrix-variate normal distribution.

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
        try:
            r_inv, _, tr_r = self.calc_r_inv_reml()
            w_inv, _ = self.calc_w_inv_reml()
        except np.linalg.LinAlgError:
            grad[:] = np.inf
            return grad
        mx = self.reml_mx_data_transformed
        rmx = r_inv @ mx
        wmxr = (w_inv * mx).T @ r_inv
        long = np.einsum('ij,ji->', wmxr, mx)
        r_grad = self.calc_r_reml_grad()
        w_inv_grad, w_logdet_grad = self.calc_w_inv_reml_grad()
        n = mx.shape[1]
        m = self.num_m
        for i, (d_r, d_w, d_wd) in enumerate(zip(r_grad, w_inv_grad,
                                                 w_logdet_grad)):
            g = 0.0
            if len(d_r.shape):
                tr = np.trace(d_r)
                g += tr * long
                t = (d_w * mx).T - wmxr @ d_r
                g += tr_r * np.einsum('ij,ji->', t, rmx)
                g += n * np.einsum('ij,ji->', r_inv, d_r)
                g += m * d_wd
                g -= n * m * tr / tr_r
            grad[i] = g
        return grad

    '''
    ------------------Matrix Variate REML (II-nd stage)-----------------------
    '''

    def obj_reml2(self, x: np.ndarray):
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
            r = self.calc_r_reml2(sigma)
            r_inv, logdet_r = chol_inv2(r)
            w_inv, logdet_w = self.calc_w_reml2_inv(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        mean = self.calc_mean(m)
        center = self.mx_data_transformed - mean
        tr_r = np.trace(r)
        n, m = self.mx_data.shape
        r_center = r_inv @ center
        center_w = center @ w_inv
        tr = tr_r * np.einsum('ji,ji->', center_w, r_center)
        return tr + m * logdet_w + n * logdet_r - n * m * np.log(tr_r)

    def grad_reml2(self, x: np.ndarray):
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
            r = self.calc_r_reml2(sigma)
            r_inv = chol_inv(r)
            w_inv, _ = self.calc_w_reml2_inv(sigma)
        except np.linalg.LinAlgError:
            grad[:] = np.inf
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
        r_grad = self.calc_r_reml2_grad(sigma_grad)
        w_grad = self.calc_w_reml2_grad(sigma_grad)
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
            grad[:] = np.inf
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
        w_inv = self.calc_w_reml2_inv(sigma)[0]
        r = self.calc_r_reml2(sigma)
        r_inv = chol_inv(r)
        m, c = aux
        sigma_grad = self.calc_sigma_grad(m, c)
        mean_grad = self.calc_mean_grad(m, c)
        w_grad = self.calc_w_reml2_grad(sigma_grad)
        r_grad = self.calc_r_reml2_grad(sigma_grad)
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
            except np.linalg.LinAlgError:
                logging.warning("Fisher Information Matrix is not PD."\
                                " Moore-Penrose inverse will be used "\
                                "instead of Cholesky decomposition. See "\
                                "10.1109/TSP.2012.2208105.")
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
                    info[i, k] += prod_means[i] @ mean_grad[k]
        fim = info + np.triu(info, 1).T
        fim = fim
        if inverse:
            fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim
