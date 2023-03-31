"""
Module for metrics.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""


from __future__ import annotations

from functools import partial
from typing import Optional, Callable, Union

import numpy as np
from scipy import sparse

from tmtoolkit.utils import partial_sparse_log


def pmi(x: Union[np.ndarray, sparse.spmatrix], y: Optional[np.ndarray] = None, xy: Optional[np.ndarray] = None,
        n_total: Optional[int] = None, logfn: Callable = np.log,
        k: int = 1, alpha: float = 1.0, normalize: bool = False) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Calculate pointwise mutual information measure (PMI). You can either pass a matrix `x` which represents counts (if
    the matrix is of dtype (u)int) N_{x,y} or probabilities p(x, y) or you pass probabilities p(x), p(y), p(x, y) given
    as `x`, `y`, `xy`, or total counts `x`, `y`, `xy` and additionally `n_total`. Setting `k` > 1 gives PMI^k variants.
    Setting `normalized` to True gives normalized PMI (NPMI) as in [Bouma2009]_. See [RoleNadif2011]_ for a comparison
    of PMI variants.

    Probabilities should be such that ``p(x, y) <= min(p(x), p(y))``.

    :param x: either a matrix with probabilities p(x, y) or counts (if matrix is of dtype (u)int); for the alternative
              calling signature that requires arguments `x`, `y` and `xy`, you can pass `x` a vector of probabilities
              p(x) or vector of number of occurrences of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrences of y (interpreted as count if `n_total` is given) when using
              alternative calling signature
    :param xy: probabilities p(x, y) or count of occurrences of x *and* y (interpreted as count if `n_total` is given)
               when using alternative calling signature
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space;
                    if `x` is given as matrix you can set `n_total` to 1 indicate that `x` is a matrix of counts even
                    if it is of dtype (u)int
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param alpha: calculate ``p_{alpha}(y)`` as ``y^alpha/sum(y^alpha)`` (only if given as counts)
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same shape as inputs containing (N)PMI measures for each input probability
    """
    def _sparse_compare(x):
        if isinstance(x, sparse.spmatrix):
            return x.nnz > 0
        else:
            return x

    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if alpha <= 0.0:
        raise ValueError('`alpha` must be strictly positive')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    if x.ndim == 1:
        if y is None or xy is None or x.shape != y.shape or x.shape != xy.shape:
            raise ValueError('if `x` is given as vector, `y` and `xy` must also be given and must have the same length')

        if alpha != 1.0:
            raise ValueError('`alpha` can only be used when `x` is passed as matrix, i.e. `alpha` '
                             'must be set to 1.0 otherwise')

        if n_total is not None:
            if n_total < 1:
                raise ValueError('`n_total` must be strictly positive')
            x = x/n_total
            y = y/n_total
            xy = xy/n_total

        # log (p(x, y) / p(x)p(y))
        logxy = logfn(xy)
        pmi_val = logxy - logfn(x) - logfn(y)
    else:
        if x.ndim != 2:
            raise ValueError('if `x` is not given as vector, it must be given as two-dimensional array')

        if y is not None or xy is not None:
            raise ValueError('if `x` is not given as vector, `y`, `xy` and `n_total` must be None')

        if x.dtype.kind in 'ui' or n_total == 1:
            # for integer matrices, we assume counts that are converted to probabilities
            # n_total == 1 is a special case to allow to pass floating point matrices as counts
            if _sparse_compare(np.any(x < 0)) or np.sum(x) == 0:
                raise ValueError('if `x` is given as matrix of counts, all elements must be positive and there must '
                                 'be at least one non-zero element')
            xy = x / np.sum(x)
            y = np.sum(xy, axis=0)

            if alpha != 1.0:
                y = np.power(y, alpha)
                y /= np.sum(y)
        else:
            if _sparse_compare(np.any(x < 0)) or _sparse_compare(np.any(x > 1)) or not np.isclose(np.sum(x), 1.0):
                raise ValueError('if `x` is given as matrix of probabilities, all elements must be in range [0, 1] and '
                                 'must sum up to 1.0')

            if alpha != 1.0:
                raise ValueError('`alpha` can only be used when `x` is passed as matrix of counts (with '
                                 'dtype (unsigned) integer), i.e. `alpha` must be set to 1.0 otherwise')

            xy = x

        x = np.sum(xy, axis=1)    # marginal prob. for columns
        if y is None:
            y = np.sum(xy, axis=0)    # marginal prob. for rows ("context")

        logfn_dense = logfn
        partial_log = False
        if isinstance(xy, sparse.spmatrix):
            if logfn is partial_sparse_log:
                logfn_dense = np.log
                partial_log = True
            elif isinstance(logfn, partial) and logfn.func is partial_sparse_log:
                logfn_dense = logfn.keywords['logfn']
                partial_log = True
            else:
                xy = xy.A  # we must convert to dense array for log
            x = x.A.flatten()
            y = y.A.flatten()

        # log (p(x, y) / p(x)p(y))
        logxy = logfn(xy)
        if partial_log:
            z = np.outer(x, y)
            xy_coo = xy.tocoo()  # to get indices of non-zero values
            # generate a matrix z where log is only applied to elements at the same position as non-zero elements in xy
            z = sparse.coo_matrix((logfn_dense(z[xy_coo.row, xy_coo.col]), (xy_coo.row, xy_coo.col)), shape=z.shape)
            pmi_val = logxy - z
        else:
            pmi_val = logxy - logfn_dense(np.outer(x, y))

    if k > 1:
        return pmi_val - (1-k) * logxy
    else:
        if normalize:
            return pmi_val / -logxy
        else:
            return pmi_val


npmi = partial(pmi, k=1, normalize=True)
pmi2 = partial(pmi, k=2, normalize=False)
pmi3 = partial(pmi, k=3, normalize=False)


def ppmi(x: Union[np.ndarray, sparse.spmatrix], y: Optional[np.ndarray] = None, xy: Optional[np.ndarray] = None,
         n_total: Optional[int] = None, logfn: Callable = np.log, add_k_smoothing: float = 0.0, alpha: float = 1.0) \
        -> Union[np.ndarray, sparse.spmatrix]:
    """
    Calculate *positive* pointwise mutual information measure (PPMI) as ``max(pmi(...), 0)``. This results in a measure
    that is in range ``[0, +Inf]``. See :func:`pmi` for further information. See [JurafskyMartin2023]_, p. 117 for more
    on (positive) PMI.

    .. note:: If you pass `x` as sparse matrix, the calculations are applied only to non-zero elements and will return
              another sparse matrix.

    :param x: either a matrix with probabilities p(x, y) or counts (if matrix is of dtype (u)int); for the alternative
              calling signature that requires arguments `x`, `y` and `xy`, you can pass `x` a vector of probabilities
              p(x) or vector of number of occurrences of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrences of y (interpreted as count if `n_total` is given) when using
              alternative calling signature
    :param xy: probabilities p(x, y) or count of occurrences of x *and* y (interpreted as count if `n_total` is given)
               when using alternative calling signature
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space;
                    if `x` is given as matrix you can set `n_total` to 1 indicate that `x` is a matrix of counts even
                    if it is of dtype (u)int
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param add_k_smoothing: can only be used when `x` is a matrix of counts; in this case `add_k_smoothing` is
                            added to `x`
    :param alpha: calculate ``p_{alpha}(y)`` as ``y^alpha/sum(y^alpha)`` (only if given as counts)
    :return: array with same shape as inputs containing PPMI measures for each input probability
    """

    input_sparse_fmt = None

    if x.ndim == 1:
        if add_k_smoothing != 0.0:
            raise ValueError('`add_k_smoothing` can only be used when `x` is passed as matrix, i.e. `add_k_smoothing` '
                             'must be set to 0.0 otherwise')

        pmi_val = pmi(x, y, xy, n_total=n_total, logfn=logfn, k=1, alpha=alpha, normalize=False)
    else:
        if isinstance(x, sparse.spmatrix):
            if add_k_smoothing != 0.0:
                # `add_k_smoothing` can only be used when `x` is **not** a sparse matrix
                x = x.A
            else:
                input_sparse_fmt = x.getformat()
                logfn = partial(partial_sparse_log, logfn=logfn)  # with this, we can retain the sparse matrix

        if x.dtype.kind in 'ui' and add_k_smoothing != 0.0:
            if isinstance(add_k_smoothing, int):
                x += add_k_smoothing
            else:
                x = x.astype('float') + add_k_smoothing

            n_total = 1   # to indicate we pass a floating point matrix with counts
        else:
            n_total = None
            if add_k_smoothing != 0.0:
                raise ValueError('`add_k_smoothing` can only be used when `x` is passed as matrix of counts (with '
                                 'dtype (unsigned) integer), i.e. `add_k_smoothing` must be set to 0.0 otherwise')

        pmi_val = pmi(x, n_total=n_total, logfn=logfn, k=1, alpha=alpha, normalize=False)

    if input_sparse_fmt is None:
        return np.maximum(pmi_val, 0)
    else:
        pmi_val = pmi_val.tocoo()
        pmi_val.data = np.maximum(pmi_val.data, 0)
        pmi_val.eliminate_zeros()
        return pmi_val.asformat(input_sparse_fmt)
