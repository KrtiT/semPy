"""Different utility functions for internal usage."""
import scipy.linalg.lapack as lapack
import pandas as pd
import numpy as np


def kron_identity(mx: np.ndarray, sz: int, back=False):
    """
    Calculate Kronecker product with identity matrix.

    Simulates np.kron(mx, np.identity(sz)).
    Parameters
    ----------
    mx : np.ndarray
        Matrix.
    sz : int
        Size of identity matrix.
    back : bool, optional
        If True, np.kron(np.identity(sz), mx) will be calculated instead. The
        default is False.

    Returns
    -------
    np.ndarray
        Kronecker product of mx and an indeity matrix.

    """
    m, n = mx.shape
    r = np.arange(sz)
    if back:
        out = np.zeros((sz, m, sz, n), dtype=mx.dtype)
        out[r,:,r,:] = mx
    else:
        out = np.zeros((m, sz, n, sz), dtype=mx.dtype)
        out[:,r,:,r] = mx
    out.shape = (m * sz,n * sz)
    return out

def delete_mx(mx: np.ndarray, exclude: np.ndarray):
    """
    Remove column and rows from square matrix.

    Parameters
    ----------
    mx : np.ndarray
        Square matrix.
    exclude : np.ndarray
        List of indices corresponding to rows/cols.

    Returns
    -------
    np.ndarray
        Square matrix without certain rows and columns.

    """
    return np.delete(np.delete(mx, exclude, axis=0), exclude, axis=1)


def cov(x: np.ndarray):
    """
    Compute covariance matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cov = np.ma.cov(masked_x, bias=True, rowvar=False).data
    if cov.size == 1:
        cov.resize((1,1))
    return cov


def cor(x: np.ndarray):
    """
    Compute correlation matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cor = np.ma.corrcoef(masked_x, bias=True, rowvar=False).data
    if cor.size == 1:
        cor.resize((1,1))
    return cor


def chol_inv(x: np.array):
    """
    Calculate invserse of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.array
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    lapack.dpotri(c, overwrite_c=1)
    c += c.T
    np.fill_diagonal(c, c.diagonal() / 2)
    return c


def chol_inv2(x: np.ndarray):
    """
    Calculate invserse and logdet of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.ndarray
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).
    logdet : float
        ln|x|

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    logdet = 2 * np.sum(np.log(c.diagonal()))
    lapack.dpotri(c, overwrite_c=1)
    c += c.T
    np.fill_diagonal(c, c.diagonal() / 2)
    return c, logdet


def compare_results(model, true: pd.DataFrame, error='relative',
                    ignore_cov=True):
    """
    Compare parameter estimates in model to parameter values in a DataFrame.

    Parameters
    ----------
    model : Model
        Model instance.
    true : pd.DataFrame
        DataFrame with operations and expected estimates. Should have "lval",
        "op", "rval", "Value" columns in this particular order.
    error : str, optional
        If 'relative', relative errors are calculated. Absolute errors are
        calculated otherwise. The default is 'relative'.
    ignore_cov : bool, optional
        If True, then covariances (~~) are ignored. The default is False.

    Raises
    ------
    Exception
        Rise when operation present in true is not present in the model.

    Returns
    -------
    errs : list
        List of errors.

    """
    ins = model.inspect(information=None)
    errs = list()
    for row in true.iterrows():
        lval, op, rval, value = row[1].values[:4]
        if op == '~~' and ignore_cov:
            continue
        if op == '=~':
            op = '~'
            lval, rval = rval, lval
        est = ins[(ins.lval == lval) & (ins.op == op) & (ins.rval == rval)]
        if len(est) == 0:
            raise Exception(f'Unknown estimate: {row}.')
        est = est.Estimate.values[0]
        if error == 'relative':
            errs.append(abs((value - est) / est))
        else:
            errs.append(abs(value - est))
    return errs


def calc_zkz(groups: pd.Series, k: pd.DataFrame):
    """
    Calculate ZKZ^T relationship matrix from covariance matrix K.

    Parameters
    ----------
    groups : pd.Series
        Series of group names for individuals.
    k : pd.DataFrame
        Covariance-across-groups matrix. If None, then its calculate as an
        identity matrix.

    Raises
    ------
    Exception
        Incorrect number of groups: mismatch between dimensions of K and
        groups series.
    KeyError
        Incorrect group naming.

    Returns
    -------
    np.ndarray
        ZKZ^T matrix.

    """
    
    p_names = list(groups.unique())
    p, n = len(p_names), len(groups)
    if k is None:
        k = np.identity(p)
    elif k.shape[0] != p:
        raise Exception("Dimensions of K don't match number of groups.")
    z = np.zeros((n, p))
    for i, germ in enumerate(groups):
        j = p_names.index(germ)
        z[i, j] = 1.0
    if type(k) is pd.DataFrame:
        try:
            k = k.loc[p_names, p_names].values
        except KeyError:
            raise KeyError("Certain groups in K differ from those "
                           "provided in a dataset.")
    zkz = z @ k @ z.T
    return zkz
        