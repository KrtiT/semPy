#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semopy 2.0 model module without random effects."""
from statsmodels.stats.correlation_tools import cov_nearest
from .utils import chol_inv, chol_inv2, cov, delete_mx
from itertools import combinations, chain
from .constraints import parse_constraint
from collections import defaultdict
from dataclasses import dataclass
from .model_base import ModelBase
from .polycorr import hetcor
from .solver import Solver
from . import startingvalues
import pandas as pd
import numpy as np
import logging


class Model(ModelBase):
    """Model class without mean-structure."""

    symb_starting_values = 'START'
    symb_bound_parameters = 'BOUND'
    symb_constraint = 'CONSTRAINT'

    # Make sure that approriate function build_matrixname() is present.
    matrices_names = 'beta', 'lambda', 'psi', 'theta'

    @dataclass
    class ParameterLoc:
        """Structure for keeping track of parameter's location in matrices."""

        __slots__ = ['matrix', 'indices', 'symmetric']

        matrix: np.ndarray
        indices: tuple
        symmetric: bool

    @dataclass
    class Parameter:
        """Structure for basic parameter info used internally in Model."""

        __slots__ = ['start', 'active', 'bound', 'locations']

        start: float
        active: bool
        bound: tuple
        locations: list

    def __init__(self, description: str, mimic_lavaan=False, baseline=False):
        """
        Instantiate Model without mean-structure.

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

        Returns
        -------
        None.

        """
        self.mimic_lavaan = mimic_lavaan
        self.parameters = dict()
        self.n_param_reg = 0
        self.n_param_cov = 0
        self.baseline = baseline
        self.constraints = list()
        dops = self.dict_operations
        dops[self.symb_starting_values] = self.operation_start
        dops[self.symb_bound_parameters] = self.operation_bound
        dops[self.symb_constraint] = self.operation_constraint
        self.objectives = {'MLW': (self.obj_mlw, self.grad_mlw),
                           'ULS': (self.obj_uls, self.grad_uls),
                           'GLS': (self.obj_gls, self.grad_gls),
                           'FIML': (self.obj_fiml, self.grad_fiml)}
        super().__init__(description)

    def add_param(self, name: str, active: bool, start: float, bound: tuple,
                  matrix: np.ndarray, indices: tuple, symmetric: bool):
        """
        Add parameter/update parameter locations in semopy matrices.

        If name is not present in self.parameters, then just locations will be
        updated. Otherwise, a new Parameter instance is added to
        self.parameters.
        Parameters
        ----------
        name : str
            Name of parameter.
        active : bool
            Is the parameter "active", i.e. is it subject to further
            optimization.
        start : float
            Starting value of parameter. If parameter is not active, then it is
            a fixing value.
        bound : tuple
            Bound constraints on parameter (a, b). None is treated as infinity.
        matrix : np.ndarray
            Reference to matrix.
        indices : tuple
            Indices of parameter in the matrix.
        symmetric : bool
            Should be True if matrix is symmetric.

        Returns
        -------
        None.

        """
        loc = self.ParameterLoc(matrix=matrix, indices=indices,
                                symmetric=symmetric)
        if name in self.parameters:
            p = self.parameters[name]
            p.locations.append(loc)
        else:
            p = self.Parameter(active=active, start=start, bound=bound,
                               locations=[loc])
            self.parameters[name] = p

    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        active_params = {name: param
                         for name, param in self.parameters.items()
                         if param.active}
        param_vals = [None] * len(active_params)
        diff_matrices = [None] * len(active_params)
        ranges = [[list(), list()] for _ in self.matrices]
        for i, (_, param) in enumerate(active_params.items()):
            param_vals[i] = param.start
            dm = [0] * len(self.matrices)
            for loc in param.locations:
                n = next(n for n in range(len(self.matrices))
                         if self.matrices[n] is loc.matrix)
                ranges[n][0].append(loc.indices)
                ranges[n][1].append(i)
                t = np.zeros_like(loc.matrix)
                t[loc.indices] = 1.0
                if loc.symmetric:
                    rind = loc.indices[::-1]
                    ranges[n][0].append(rind)
                    ranges[n][1].append(i)
                    t[rind] = 1.0
                dm[n] += t
            diff_matrices[i] = [m if type(m) is not int else None for m in dm]
        for rng in ranges:
            rng[0] = tuple(zip(*rng[0]))
        self.param_vals = np.array(param_vals)
        self.param_ranges = ranges
        self.mx_diffs = diff_matrices
        self.identity_c = np.identity(self.mx_beta.shape[0])

    def post_classification(self, effects: dict):
        """
        Procedure that is run just after classify_variables.

        Parameters
        -------
        effects : dict
            Maping opcode->values->rvalues->mutiplicator.

        Returns
        -------
        None.

        """
        self.finalize_variable_classification()
        self.preprocess_effects(effects)
        self.setup_matrices()

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        cov = effects[self.symb_covariance]
        exo = self.vars['exogenous']
        obs_exo = set(self.vars['observed']) & exo
        for v in obs_exo:
            if v not in cov[v]:
                cov[v][v] = self.symb_starting_values
        for a, b in combinations(obs_exo, 2):
            if a not in cov[b] and b not in cov[a]:
                cov[a][b] = self.symb_starting_values
        for v in chain(self.vars['endogenous'], self.vars['latent']):
            if v not in cov[v]:
                cov[v][v] = None
        exo_lat = self.vars['exogenous'] & self.vars['latent']
        for a, b in chain(combinations(self.vars['output'], 2),
                          combinations(exo_lat, 2)):
            if b not in cov[a] and a not in cov[b]:
                cov[a][b] = None

    def finalize_variable_classification(self):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.
        Returns
        -------
        None.

        """
        if not self.mimic_lavaan:
            outputs = self.vars['output']
            self.vars['_output'] = outputs
            self.vars['output'] = {}
        else:
            inds = self.vars['indicator']
            # It's possible that user violates that classical requirement that
            # measurement variables must be outputs, here we take such case
            # in account by ANDing inds with outputs.
            self.vars['_output'] = inds & self.vars['output']
            self.vars['output'] -= inds
        inners = self.vars['all'] - self.vars['_output']
        self.vars['inner'] = inners
        # This is entirely for visual reasons only, one might replace sorted
        # with list as well.
        obs = self.vars['observed']
        inners = sorted(self.vars['latent'] & inners) + sorted(obs & inners)
        self.vars['inner'] = inners
        t = obs & self.vars['_output']
        self.vars['observed'] = sorted(t) + sorted(obs - t)

    def setup_matrices(self):
        """
        Initialize base matrix structures of the model.

        Returns
        -------
        None.

        """
        # I don't use dicts here as matrices structures and further ranges
        # structures will be used very often in optimsation procedure, and
        # key-retrieival process takes a toll on performance.
        # Please, don't rearrange matrices below, let the order stay the same.
        # It's necessary because I assumed fixed matrix positions in those
        # structures when coding linear algebra parts of the code.
        self.matrices = list()
        self.names = list()
        self.start_rules = list()
        for v in self.matrices_names:
            mx, names = getattr(self, f'build_{v}')()
            setattr(self, f'mx_{v}', mx)
            setattr(self, f'names_{v}', names)
            self.matrices.append(mx)
            self.names.append(names)
            self.start_rules.append(getattr(startingvalues, f'start_{v}'))

    def build_beta(self):
        """
        Beta matrix contains relationships between all non-_output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['inner']
        n = len(names)
        mx = np.zeros((n, n))
        return mx, (names, names)

    def build_lambda(self):
        """
        Lambda matrix loads non-_output variables onto _output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        obs = self.vars['observed']
        inner = self.vars['inner']
        row, col = obs, inner
        n, m = len(row), len(col)
        mx = np.zeros((n, m))
        for v in obs:
            if v in inner:
                i, j = obs.index(v), inner.index(v)
                mx[i, j] = 1.0
        return mx, (row, col)

    def build_psi(self):
        """
        Psi matrix is a covariance matrix for non-_output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['inner']
        n = len(names)
        mx = np.zeros((n, n))
        return mx, (names, names)

    def build_theta(self):
        """
        Theta matrix is a covariance matrix for _output variables.

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

    def effect_regression(self, items: dict):
        """
        Work through regression operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        if self.baseline:
            return
        outputs = self.vars['_output']
        for lv, rvs in items.items():
            lv_is_out = lv in outputs
            if lv_is_out:
                mx = self.mx_lambda
                rows, cols = self.names_lambda
            else:
                mx = self.mx_beta
                rows, cols = self.names_beta
            i = rows.index(lv)
            for rv, mult in rvs.items():
                name = None
                active = True
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    if mult is not None:
                        if mult == self.symb_starting_values:
                            active = False
                        else:
                            name = mult
                    val = None
                if name is None:
                    self.n_param_reg += 1
                    name = '_b%s' % self.n_param_reg
                j = cols.index(rv)
                ind = (i, j)
                self.add_param(name=name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=False,
                               bound=(None, None))

    def effect_measurement(self, items: dict):
        """
        Work through measurement operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Raises
        -------
        Exception
            Rises when indicator is misspecified and not observable.

        Returns
        -------
        None.

        """
        reverse_dict = defaultdict(dict)
        self.first_manifs = defaultdict(lambda: None)
        obs = self.vars['observed']
        for lat, inds in items.items():
            first = None
            lt = list()
            for ind, mult in inds.items():
                if ind not in obs:
                    logging.info(f'Manifest variables should be observed,\
                                   but {ind} appears to be latent.')
                try:
                    float(mult)
                    if first is None:
                        first = len(lt)
                except (TypeError, ValueError):
                    if mult == self.symb_starting_values and first is None:
                        first = len(lt)
                lt.append((ind, mult))
            if first is None:
                for i, (ind, mult) in enumerate(lt):
                    if mult is None:
                        first = i
                        break
                if first is None:
                    logging.warning('No fixed loadings for %s.', lat)
                else:
                    lt[first] = (ind, 1.0)
            for ind, mult in lt:
                reverse_dict[ind][lat] = mult
            if first is not None:
                self.first_manifs[lat] = lt[first][0]
        self.effect_regression(reverse_dict)

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

    def inspect(self, mode='list', what='est', information='expected',
                std_est=False):
        """
        Get fancy view of model parameters estimates.
    
        Parameters
        ----------
        model : str
            Model.
        mode : str, optional
            If 'list', pd.DataFrame with estimates and p-values is returned.
            If 'mx', a dictionary of matrices is returned. The default is 'list'.
        what : TYPE, optional
            Used only if mode == 'mx'. If 'est', matrices have estimated
            values. If 'start', matrices have starting values. If 'name', matrices
            have names inplace of their parameters. The default is 'est'.
        information : str
            If 'expected', expected Fisher information is used. Otherwise,
            observed information is employed. No use if mode is not 'list'.
            The default is 'expected'.
        std_est : bool
            If True, standardized coefficients are also returned as Std. Ests
            col. If it is 'lv', then output variables are not standardized.
            The default is False.

        Returns
        -------
        pd.DataFrame | dict
            Dataframe or mapping matrix_name->matrix.
    
        """
        from .inspector import inspect
        return inspect(self, mode=mode, what=what, information=information,
                       std_est=std_est)

    def operation_start(self, operation):
        """
        Works through START command.

        Sets starting values to parameters.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Raise
        ----------
        IndexError
            When no starting value is supplied as an argument to START.
        KeyError
            When invalid parameter name is supplied.

        Returns
        -------
        None.

        """
        try:
            start = operation.params[0]
        except IndexError:
            raise IndexError('START must have starting value as an argument.')
        for param in operation.onto:
            try:
                self.parameters[param].start = start
            except KeyError:
                raise KeyError(f'{param} is not a valid parameter name.')

    def operation_bound(self, operation):
        """
        Works through BOUND command.

        Sets bound constraints to parameters.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Raise
        ----------
        SyntaxError
            When no starting value is supplied as an argument to START.
        ValueError
            When BOUND arguments are not translatable through FLOAT.
        KeyError
            When invalid parameter name is supplied.

        Returns
        -------
        None.

        """
        try:
            a = float(operation.params[0])
            b = float(operation.params[1])
            b = (a, b)
        except IndexError:
            raise SyntaxError('BOUND must have 2 bounding arguments.')
        except ValueError:
            raise ValueError('BOUND arguments must be floats.')
        for param in operation.onto:
            try:
                self.parameters[param].bound = b
            except KeyError:
                raise KeyError(f'{param} is not a valid parameter name.')

    def operation_constraint(self, operation):
        """
        Works through CONSTRAINT command.

        Adds inequality and equality constraints.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Raise
        ----------
        SyntaxError
            When no starting value is supplied as an argument to START.

        Returns
        -------
        None.

        """
        try:
            constr = operation.params[0]
        except IndexError:
            raise SyntaxError('CONSTRAINT must have 1 argument: constraint.')
        params = [name for name, param in self.parameters.items()
                  if param.active]
        self.constraints.append(parse_constraint(constr, params))

    def operation_define(self, operation):
        """
        Works through DEFINE command.

        Here, used to add ordinal variables to the variable holder.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        if operation.params and operation.params[0] == 'ordinal':
            if 'ordinal' not in self.vars:
                ords = set()
                self.vars['ordinal'] = ords
            else:
                ords = self.vars['ordinal']
            ords.update(operation.onto)  

    def update_matrices(self, params: np.ndarray):
        """
        Update all matrices from a parameter vector.

        Parameters
        ----------
        params : np.ndarray
            Vector of parameters.

        Returns
        -------
        None.

        """
        for mx, (r1, r2) in zip(self.matrices, self.param_ranges):
            if r1:
                mx[r1] = params[r2]

    def load_starting_values(self):
        """
        Load starting values for parameters from empirical data.

        Returns
        -------
        None.

        """
        for _, param in self.parameters.items():
            if param.start is None:
                loc = param.locations[0]
                n = next(n for n in range(len(self.matrices))
                         if self.matrices[n] is loc.matrix)
                row, col = self.names[n]
                lval, rval = row[loc.indices[0]], col[loc.indices[1]]
                param.start = self.start_rules[n](self, lval, rval)
            for loc in param.locations:
                loc.matrix[loc.indices] = param.start
                if loc.symmetric:
                    loc.matrix[loc.indices[::-1]] = param.start

    def load_data(self, data: pd.DataFrame, covariance=None, groups=None):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        covariance : pd.DataFrame, optional
            Custom covariance matrix. The default is None.
        groups : list, optional
            List of group names to center across. The default is None.

        Returns
        -------
        None.

        """
        if groups is None:
            groups = list()
        obs = self.vars['observed']
        for group in groups:
            for g in data[group].unique():
                inds = data[group] == g
                if sum(inds) == 1:
                    continue
                data.loc[inds, obs] -= data.loc[inds, obs].mean()
                data.loc[inds, group] = g
        self.mx_data = data[obs].values
        if len(self.mx_data.shape) != 2:
            self.mx_data = self.mx_data[:, np.newaxis]
        if 'ordinal' not in self.vars:
            self.load_cov(covariance.loc[obs, obs].values
                          if covariance is not None else cov(self.mx_data))
        else:
            inds = [obs.index(v) for v in self.vars['ordinal']]
            self.load_cov(hetcor(self.mx_data, inds))

    def load_cov(self, covariance: np.ndarray):
        """
        Load covariance matrix.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix.

        Returns
        -------
        None.

        """
        if type(covariance) is pd.DataFrame:
            obs = self.vars['observed']
            covariance = covariance.loc[obs, obs].values
        if covariance.size == 1:
            covariance.resize((1, 1))
        self.mx_cov = covariance
        try:
            self.mx_cov_inv, self.cov_logdet = chol_inv2(self.mx_cov)
        except np.linalg.LinAlgError:
            logging.warning('Sample covariance matrix is not PD. It may '
                            'indicate that data is bad. Also, it arises often '
                            'when polychoric/polyserial correlations are used.'
                            ' semopy now will run nearPD subroutines.')
            self.mx_cov = cov_nearest(covariance, threshold=1e-2)
            self.mx_cov_inv, self.cov_logdet = chol_inv2(self.mx_cov)
        self.mx_covlike_identity = np.identity(self.mx_cov.shape[0])

    def get_bounds(self):
        """
        Get bound constraints on parameters.

        Returns
        -------
        list
            List of tuples specifying bounds.

        """
        return [param.bound for _, param in self.parameters.items()
                if param.active]

    def load(self, data=None, cov=None, groups=None, clean_slate=False):
        """
        Load dataset.

        Either data or cov must be provided. If both are provided, covariance
        matrix is not estimated from data, but data is used for starting values
        initialization. If only cov is provided, it's impossible to estimate
        starting values for regression coefficients, also attempts to estimate
        polychoric/polyserial correlations if requested are ignored.
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        groups : list, optional
            Groups of size > 1 to center across. The default is None.
        clean_slate : bool, optional
            If True, resets parameters vector. The default is False.

        Raises
        ------
        Exception
            Rises when both data and cov are not provided or when FIML is used
            in abscence of full data.

        KeyError
            Rises when there are missing variables from the data.

        Returns
        -------
        None.

        """
        if data is None and cov is None:
            has_data = hasattr(self, 'mx_data')
            has_cov = hasattr(self, 'mx_cov')
            if not has_data and not has_cov:
                raise Exception('Either data or cov must be provided.')
            return
        if data is None:
            logging.info('Providing only covariance matrix is unadvised as it\
                         prevents from estimating good starting values for\
                         loadings.')
        else:
            data = data.copy()
        cols = data.columns if data is not None else cov.columns
        obs = self.vars['observed']
        missing = set(obs) - set(set(cols))
        if missing:
            t = ', '.join(missing)
            raise KeyError('Variables {} are missing from data.'.format(t))
        if data is not None:
            self.load_data(data, covariance=cov, groups=groups)
        else:
            self.load_cov(cov)
            if groups is not None:
                logging.warning('"groups" argument is redunant with cov \
                                matrix.')
        if data is not None or cov is not None:
            self.load_starting_values()
        if clean_slate or not hasattr(self, 'param_vals'):
            self.prepare_params()

    def fit(self, data=None, cov=None, obj='MLW', solver='SLSQP', groups=None,
            clean_slate=False, regularization=None):
        """
        Fit model to data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        obj : str, optional
            Objective function to minimize. Possible values are 'MLW', 'FIML',
            'ULS', 'GLS'. The default is 'MLW'.
        solver : TYPE, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        groups : list, optional
            Groups of size > 1 to center across. The default is None.
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
            Rises when attempting to use FIML in absence of full data.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        self.load(data=data, cov=cov, groups=groups,
                  clean_slate=clean_slate)
        if obj == 'FIML':
            if not hasattr(self, 'mx_data'):
                raise Exception('Full data must be supplied for FIML')
            self.prepare_fiml()
        fun, grad = self.get_objective(obj, regularization=regularization)
        solver = Solver(solver, fun, grad, self.param_vals,
                        constrs=self.constraints,
                        bounds=self.get_bounds())
        res = solver.solve()
        res.name_obj = obj
        self.param_vals = res.x
        self.update_matrices(res.x)
        self.last_result = res
        return res

    def get_objective(self, name: str, regularization=None):
        """
        Retrieve objective function and its gradient by name.

        Parameters
        ----------
        name : str
            Name of objective function.
        regularization
            Special structure as returend by create_regularization function.
            If not None, then a regularization will be applied to a certain
            parameters in the model. The default is None.

        Raises
        ------
        KeyError
            Rises if incorrect name is provided.

        Returns
        -------
        tuple
            Objective function and gradient function.

        """
        try:
            if regularization is None:
                return self.objectives[name]
            else:
                fun, grad = self.objectives[name]
                regu, regu_grad = regularization
                if regu_grad is None:
                    return lambda x: fun(x) + regu(x), None
                else:
                    return lambda x: fun(x) + regu(x),\
                        lambda x: grad(x) + regu_grad(x)
        except KeyError:
            raise KeyError(f'{name} is unknown objective function.')

    def prepare_fiml(self):
        """
        Prepare data structure for efficient FIML calculation.

        Returns
        -------
        None.

        """
        d = defaultdict(list)
        for i in range(self.mx_data.shape[0]):
            t = tuple(list(np.where(np.isfinite(self.mx_data[i]))[0]))
            d[t].append(i)
        for cols, rows in d.items():
            inds = tuple(i for i in range(self.mx_data.shape[1])
                         if i not in cols)
            t = self.mx_data[rows][:, cols]
            d[cols] = (t.T @ t, inds, len(rows))
        self.fiml_data = d

    def predict(self, x: pd.DataFrame, solver='SLSQP', factors=True,
                ret_opt=False):
        """
        Predict/impute data given certain observations.

        Parameters
        ----------
        x : pd.DataFrame
            Observations with possibly missing entries of certain variables.
            It's possible to provide latent variables "observations" too.
        solver : str, optional
            Solver to use. The default is 'SLSQP'.
        factors: bool
            If True, factor scores are estimated. The default is True.
        ret_opt : SolverResult, optional
            If True, SolverResult is also returned. The default is False.

        Returns
        -------
        pd.DataFrame
            Table with imputed data.

        """
        from .imputer import get_imputer
        imp = get_imputer(self)(self, x, factors=factors)
        res = imp.fit(solver='SLSQP')
        data = imp.get_fancy()
        return data if not ret_opt else (data, res)

    '''
    ----------------------------LINEAR ALGEBRA PART---------------------------
    ----------------------The code below is responsible-----------------------
    ------------------for covariance structure computations-------------------
    '''

    def calc_sigma(self):
        """
        Calculate model-implied covariance matrix.

        Returns
        -------
        sigma : np.ndarray
            Sigma model-implied covariance matrix.
        tuple
            Tuple of auxiliary matrics Lambda @ C and C, where C = (I - B)^-1.

        """
        beta, lamb, psi, theta = self.matrices[:4]
        c = np.linalg.inv(self.identity_c - beta)
        m = lamb @ c
        return m @ psi @ m.T + theta, (m, c)

    def calc_sigma_grad(self, m: np.ndarray, c: np.ndarray):
        """
        Calculate gradient wrt to parameters vector of Sigma matrix.

        Parameters
        ----------
        m : np.ndarray
            Auxilary matrix returned from calc_sigma Lambda @ C.
        c : np.ndarray
            Auxilary matrix returned from calc_sigma (I - B)^-1

        Returns
        -------
        grad : list
            List of derivatives of Sigma wrt to parameters vector.

        """
        psi = self.matrices[2]
        m_t = m.T
        p = c @ psi
        d = p @ m_t
        grad = list()
        for dmxs in self.mx_diffs:
            g = np.float32(0.0)
            if dmxs[0] is not None:  # Beta
                t = dmxs[0] @ p
                g += m @ (t + t.T) @ m_t
            if dmxs[1] is not None:  # Lambda
                t = dmxs[1] @ d
                g += t + t.T
            if dmxs[2] is not None:  # Psi
                g += m @ dmxs[2] @ m_t
            if dmxs[3] is not None:  # Theta
                g += dmxs[3]
            grad.append(g)
        return grad

    '''
    ----------------------The code below is responsible-----------------------
    ------------------for objective function computations---------------------
    '''

    '''
    ------------------------Wishart logratio function-------------------------
    '''

    def obj_mlw(self, x: np.ndarray):
        """
        Calculate Wishart likelihood logratio.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            Logratio of likelihoods.

        """
        self.update_matrices(x)
        try:
            sigma, _ = self.calc_sigma()
            inv_sigma, logdet_sigma = chol_inv2(sigma)
        except np.linalg.LinAlgError:
            return np.inf
        log_det_ratio = logdet_sigma - self.cov_logdet
        tr = np.einsum('ij,ji->', self.mx_cov, inv_sigma) - sigma.shape[0]
        loss = tr + log_det_ratio
        if loss < 0:  # Realistically should never happen.
            return np.inf
        return loss

    def grad_mlw(self, x: np.ndarray):
        """
        Calculate Wishart likelihood logratio gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of MLW.

        """
        self.update_matrices(x)
        try:
            sigma, (m, c) = self.calc_sigma()
            inv_sigma = chol_inv(sigma)
        except np.linalg.LinAlgError:
            t = np.zeros((len(x),))
            t[:] = np.inf
            return t
        sigma_grad = self.calc_sigma_grad(m, c)
        cs = inv_sigma - inv_sigma @ self.mx_cov @ inv_sigma
        return np.array([np.einsum('ij,ji->', cs, g)
                         for g in sigma_grad])

    '''
    ---------------------Full Information Maximum Likelihood-------------------
    '''

    def obj_fiml(self, x: np.ndarray):
        """
        Calculate FIML objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            FIML.

        """
        self.update_matrices(x)
        sigma_full, _ = self.calc_sigma()
        tr = 0
        logdet = 0
        for _, (mx, inds, n) in self.fiml_data.items():
            sigma = delete_mx(sigma_full, inds)
            try:
                sigma_inv, logdet_sigma = chol_inv2(sigma)
            except np.linalg.LinAlgError:
                return np.inf
            tr += np.einsum('ij,ji->', mx, sigma_inv)
            logdet += n * logdet_sigma
        loss = tr + logdet
        if loss < 0:  # Realistically should never happen.
            return np.inf
        return loss

    def grad_fiml(self, x: np.ndarray):
        """
        Calculate FIML gradient.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of FIML.

        """
        self.update_matrices(x)
        sigma_full, (m, c) = self.calc_sigma()
        sigma_grad = self.calc_sigma_grad(m, c)
        grad = [0.0] * len(sigma_grad)
        for _, (mx, inds, n) in self.fiml_data.items():
            sigma = delete_mx(sigma_full, inds)
            try:
                sigma_inv = chol_inv(sigma)
            except np.linalg.LinAlgError:
                t = np.zeros(len(grad))
                t[:] = np.inf
                return t
            t = sigma_inv @ mx @ sigma_inv
            for i, g in enumerate(sigma_grad):
                g = delete_mx(g, inds)
                tr = -np.einsum('ij,ji->', t, g)
                logdet = n * np.einsum('ij,ji->', sigma_inv, g)
                grad[i] += tr + logdet
        return np.array(grad)

    '''
    -------------------------Unweighted Least Squares-------------------------
    '''

    def obj_uls(self, x: np.ndarray):
        """
        Calculate ULS objective value.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            ULS value.

        """
        self.update_matrices(x)
        try:
            sigma, _ = self.calc_sigma()
        except np.linalg.LinAlgError:
            return np.inf
        t = sigma - self.mx_cov
        loss = np.einsum('ij,ij->', t, t)
        return loss

    def grad_uls(self, x: np.ndarray):
        """
        Gradient of ULS objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of ULS.

        """
        self.update_matrices(x)
        try:
            sigma, (m, c) = self.calc_sigma()
        except np.linalg.LinAlgError:
            t = np.zeros((len(x),))
            t[:] = np.inf
            return t
        sigma_grad = self.calc_sigma_grad(m, c)
        t = sigma - self.mx_cov
        return 2 * np.array([np.einsum('ij,ji->', g, t)
                             for g in sigma_grad])

    '''
    -------------------------Generalized Least Squares------------------------
    '''

    def obj_gls(self, x: np.ndarray):
        """
        Calculate GLS objective value.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        float
            GLS value.

        """
        self.update_matrices(x)
        try:
            sigma, _ = self.calc_sigma()
        except np.linalg.LinAlgError:
            return np.inf
        t = sigma @ self.mx_cov_inv - self.mx_covlike_identity
        return np.einsum('ij,ji->', t, t)

    def grad_gls(self, x: np.ndarray):
        """
        Gradient of GLS objective function.

        Parameters
        ----------
        x : np.ndarray
            Parameters vector.

        Returns
        -------
        np.ndarray
            Gradient of GLS.

        """
        self.update_matrices(x)
        try:
            sigma, (m, c) = self.calc_sigma()
        except np.linalg.LinAlgError:
            t = np.zeros((len(x),))
            t[:] = np.inf
            return t
        sigma_grad = self.calc_sigma_grad(m, c)
        t = self.mx_cov_inv @ \
            (sigma @ self.mx_cov_inv - self.mx_covlike_identity)
        return 2 * np.array([np.einsum('ij,ji->', g, t)
                             for g in sigma_grad])

    '''
    -------------------------Fisher Information Matrix------------------------
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
        inv_sigma = chol_inv(sigma)
        sz = len(sigma_grad)
        info = np.zeros((sz, sz))
        sgs = [sg @ inv_sigma for sg in sigma_grad]
        n = self.mx_data.shape[0] / 2
        for i in range(sz):
            for k in range(i, sz):
                info[i, k] = n * np.einsum('ij,ji->', sgs[i], sgs[k])
        fim = info + np.triu(info, 1).T
        if inverse:
            try:
                fim_inv = chol_inv(fim)
                self._fim_warn = False
            except np.linalg.LinAlgError:
                logging.warn("Fisher Information Matrix is not PD."
                              "Moore-Penrose inverse will be used instead of "
                              "Cholesky decomposition. See "
                              "10.1109/TSP.2012.2208105.")
                self._fim_warn = True
                fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim
