"""Different utility functions for internal usage."""
import scipy.linalg.lapack as lapack
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
    return np.ma.cov(masked_x, bias=True, rowvar=False).data


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
    return np.ma.corrcoef(masked_x, bias=True, rowvar=False).data


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
