import logging
import math
import os.path
import string
from datetime import date

import pytest
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
import pandas as pd
from scipy import sparse

from ._testtools import strategy_dtm_small, strategy_2d_array

from tmtoolkit.utils import (pickle_data, unpickle_file, flatten_list, greedy_partitioning,
                             mat2d_window_from_indices, combine_sparse_matrices_columnwise, path_split, read_text_file,
                             linebreaks_win2unix, split_func_args, empty_chararray, as_chararray, merge_dicts,
                             merge_sets, sample_dict, enable_logging, set_logging_level, disable_logging, dict2df,
                             applychain, indices_of_matches, chararray_elem_size, check_context_size,
                             pairwise_max_table, partial_sparse_log)

PRINTABLE_ASCII_CHARS = [chr(c) for c in range(32, 127)]


@pytest.mark.parametrize('level, fmt', [
    (logging.DEBUG, '%(levelname)s:%(name)s:%(message)s'),
    (logging.INFO, '%(levelname)s:%(name)s:%(message)s'),
    (logging.WARNING, '%(levelname)s:%(name)s:%(message)s'),
    (logging.INFO, '<default>'),
])
def test_enable_disable_logging(caplog, level, fmt):
    tmtk_logger = logging.getLogger('tmtoolkit')
    tmtk_logger.setLevel(logging.WARNING)      # reset to default level

    tmtk_logger.debug('test line debug 1')
    tmtk_logger.info('test line info 1')
    assert caplog.text == ''

    # pytest caplog fixture uses an extra logging handler (which is already added to the logger)
    if fmt == '<default>':
        enable_logging(level, logging_handler=caplog.handler, add_logging_handler=False)
    else:
        enable_logging(level, fmt, logging_handler=caplog.handler, add_logging_handler=False)

    tmtk_logger.debug('test line debug 2')
    if level == logging.DEBUG:
        assert caplog.text.endswith('DEBUG:tmtoolkit:test line debug 2\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())
    else:
        assert caplog.text == ''

    caplog.clear()

    tmtk_logger.info('test line info 2')
    if level <= logging.INFO:
        assert caplog.text.endswith('INFO:tmtoolkit:test line info 2\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())
    else:
        assert caplog.text == ''

    if level > logging.DEBUG:   # reduce logging level to DEBUG
        caplog.clear()
        set_logging_level(logging.DEBUG)
        tmtk_logger.debug('test line debug 3')
        assert caplog.text.endswith('DEBUG:tmtoolkit:test line debug 3\n')
        if fmt == '<default>':
            assert caplog.text.startswith(date.today().isoformat())

    caplog.clear()
    disable_logging()

    tmtk_logger.debug('test line debug 4')
    tmtk_logger.info('test line info 4')

    assert caplog.text == ''


def test_pickle_unpickle():
    pfile = os.path.join('tests', 'data', 'test_pickle_unpickle.pickle')
    input_data = ('foo', 123, [])
    pickle_data(input_data, pfile)

    output_data = unpickle_file(pfile)

    for i, o in zip(input_data, output_data):
        assert i == o


def test_path_split():
    assert path_split('') == []
    assert path_split('/') == []
    assert path_split('a') == ['a']
    assert path_split('/a') == ['a']
    assert path_split('/a/') == ['a']
    assert path_split('a/') == ['a']
    assert path_split('a/b') == ['a', 'b']
    assert path_split('a/b/c') == ['a', 'b', 'c']
    assert path_split('/a/b/c') == ['a', 'b', 'c']
    assert path_split('/a/b/c/') == ['a', 'b', 'c']
    assert path_split('/a/../b/c/') == ['a', '..', 'b', 'c']
    assert path_split('/a/b/c/d.txt') == ['a', 'b', 'c', 'd.txt']


def test_read_text_file():
    fpath = os.path.join('tests', 'data', 'gutenberg', 'kafka_verwandlung.txt')
    contents = read_text_file(fpath, encoding='utf-8')
    assert len(contents) > 0
    contents = read_text_file(fpath, encoding='utf-8', read_size=10)
    assert 5 <= len(contents) <= 10
    contents = read_text_file(fpath, encoding='utf-8', read_size=10, force_unix_linebreaks=False)
    assert len(contents) == 10
    contents = read_text_file(fpath, encoding='utf-8', read_size=100)
    assert 0 < len(contents) <= 100


@given(text=st.text(alphabet=list('abc \r\n'), max_size=20))
def test_linebreaks_win2unix(text):
    res = linebreaks_win2unix(text)
    assert '\r\n' not in res
    if '\r\n' in text:
        assert '\n' in res


def test_empty_chararray():
    res = empty_chararray()
    assert isinstance(res, np.ndarray)
    assert len(res) == 0
    assert res.ndim == 1
    assert np.issubdtype(res.dtype, 'str')


@given(x=st.lists(st.integers()),
       as_numpy_array=st.booleans())
def test_as_chararray(x, as_numpy_array):
    x_orig = x
    if as_numpy_array:
        x = np.array(x)

    res = as_chararray(x)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(x)
    assert res.ndim == 1
    assert np.issubdtype(res.dtype, 'str')
    assert res.tolist() == list(map(str, x_orig))


@given(x=st.lists(st.text()),
       as_numpy_array=st.booleans())
def test_chararray_elem_size(x, as_numpy_array):
    if as_numpy_array:
        x = as_chararray(x)
        assert chararray_elem_size(x) == max(1, np.max(np.char.str_len(x))) if len(x) > 0 else 1
    else:
        with pytest.raises(ValueError):
            chararray_elem_size(x)


@given(a=arrays(int, array_shapes(min_dims=1, max_dims=1), elements=st.integers(-5, 5)),
       b=arrays(int, array_shapes(min_dims=1, max_dims=1), elements=st.integers(-5, 5), unique=True),
       b_is_sorted=st.booleans(),
       check_a_in_b=st.booleans())
def test_indices_of_matches(a, b, b_is_sorted, check_a_in_b):
    if not check_a_in_b:
        a = a[np.in1d(a, b)]

    if b_is_sorted:
        b = np.sort(b)

    if check_a_in_b and np.any(~np.in1d(a, b)):
        with pytest.raises(ValueError):
            indices_of_matches(a, b, b_is_sorted=b_is_sorted, check_a_in_b=check_a_in_b)
    else:
        ind = indices_of_matches(a, b, b_is_sorted=b_is_sorted, check_a_in_b=check_a_in_b)
        assert isinstance(ind, np.ndarray)
        assert ind.shape == a.shape

        matched = b[ind]
        assert matched.shape == a.shape
        assert np.all(matched == a)


@given(m=strategy_2d_array(int, 0, 10),
       pass_sparse=st.booleans(),
       labels=st.booleans(),
       output_columns=st.booleans(),
       sort=st.integers(0, 4),   # 0: no sort, 1: default sort, 2 to 4: column indices
       sort_asc=st.booleans(),
       skip_zeros=st.booleans())
def test_pairwise_max_table(m, pass_sparse, labels, output_columns, sort, sort_asc, skip_zeros):
    default_cols = ['x', 'y', 'value']

    if pass_sparse:
        m = sparse.csr_matrix(m)

    args = dict(skip_zeros=skip_zeros)
    n = min(m.shape)

    generated_labels = ['l' + str(i) for i in range(n)]

    if labels:
        args['labels'] = generated_labels

    if output_columns:
        args['output_columns'] = ['a', 'b', 'n']

    if sort > 2:
        if output_columns:
            args['sort'] = args['output_columns'][sort-2]
        else:
            args['sort'] = default_cols[sort-2]

        if not sort_asc:
            args['sort'] = '-' + args['sort']
    else:
        args['sort'] = bool(sort)
        if args['sort']:
            sort_asc = False

    if m.shape[0] != m.shape[1]:
        with pytest.raises(ValueError, match='^`m` must be a square matrix'):
            pairwise_max_table(m=m, **args)

    # truncate
    m = m[:n, :n]

    # run
    res = pairwise_max_table(m=m, **args)

    # check
    assert isinstance(res, pd.DataFrame)
    res_rows, res_cols = res.shape

    if n == 0:
        assert res_rows == 0

    if skip_zeros:
        assert res_rows <= n
    else:
        assert res_rows == n
    assert res_cols == 3

    if output_columns:
        assert res.columns.tolist() == args['output_columns']
    else:
        assert res.columns.tolist() == default_cols

    res_lbls = res.iloc[:, :2].to_numpy()
    res_vals = res.iloc[:, 2].to_numpy()

    if res_rows > 0:
        max_val = np.max(m)
        assert np.isin(max_val, res_vals)

        if sort == 1 or args['sort'] == res.columns.tolist()[-1]:
            if sort_asc:
                assert res_vals[-1] == max_val
            else:
                assert res_vals[0] == max_val

        if skip_zeros:
            assert np.all(res_vals > 0)

        if labels:
            assert np.issubdtype(res_lbls.dtype, 'str') or np.issubdtype(res_lbls.dtype, object)
            assert np.all(lbl in args['labels'] for lbl in res_lbls.flatten())
        else:
            assert np.issubdtype(res_lbls.dtype, 'int')
            assert np.all(res_lbls >= 0)
            assert np.all(res_lbls < n)

    # run again using dataframe as input
    df = pd.DataFrame(m.todense() if sparse.issparse(m) else m, index=generated_labels, columns=generated_labels)
    res2 = pairwise_max_table(m=df, **args)

    if labels:
        assert np.all(res == res2)
    else:
        assert np.all(res.iloc[:, 2] == res2.iloc[:, 2])


@given(data=st.dictionaries(keys=st.text(string.ascii_letters, min_size=1), values=st.integers(), max_size=10),
       key_name=st.text(string.ascii_letters, min_size=1),
       value_name=st.text(string.ascii_letters, min_size=1),
       sort=st.sampled_from([None, 'key', 'value']),
       asc=st.booleans())
def test_dict2df(data, key_name, value_name, sort, asc):
    if sort == 'key':
        sort_arg = key_name
    elif sort == 'value':
        sort_arg = value_name
    else:
        sort_arg = None

    if not asc and sort is not None:
        sort_arg = '-' + sort_arg

    if key_name == value_name:
        with pytest.raises(ValueError):
            dict2df(data, key_name, value_name, sort=sort_arg)
    else:
        res = dict2df(data, key_name, value_name, sort=sort_arg)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == len(data)
        assert res.columns.tolist() == [key_name, value_name]

        # check key - value mapping
        for k, v in data.items():
            cell = res.loc[res[key_name] == k, value_name].tolist()
            assert len(cell) == 1
            assert cell[0] == v

        # check sort
        if sort == 'key':
            assert res[key_name].tolist() == sorted(data.keys(), reverse=not asc)
        elif sort == 'value':
            assert res[value_name].tolist() == sorted(data.values(), reverse=not asc)
        else:
            assert res[key_name].tolist() == list(data.keys())
            assert res[value_name].tolist() == list(data.values())


@pytest.mark.parametrize('expected, funcs, initial_arg', [
    (None, [], 1),
    (1, [lambda x: x], 1),
    (1, [lambda x: -x, lambda x: -x], 1),
    (2.0, [lambda x: x**2, math.sqrt], 2),
    (8.0, [lambda x: x**2, math.sqrt, lambda x: x**3], 2),
])
def test_applychain(expected, funcs, initial_arg):
    if expected is None:
        with pytest.raises(ValueError):
            applychain(funcs, initial_arg)
    else:
        res = applychain(funcs, initial_arg)
        if isinstance(expected, float):
            assert math.isclose(res, expected)
        else:
            assert res == expected


@given(l=st.lists(st.integers(0, 10), min_size=2, max_size=2).flatmap(
    lambda size: st.lists(st.lists(st.integers(), min_size=size[0], max_size=size[0]),
                          min_size=size[1], max_size=size[1])))
def test_flatten_list(l):
    l_ = flatten_list(l)

    assert type(l_) is list
    assert len(l_) == sum(map(len, l))


@given(
    mat=strategy_dtm_small(),
    n_row_indices=st.integers(0, 10),
    n_col_indices=st.integers(0, 10),
    copy=st.booleans()
)
def test_mat2d_window_from_indices(mat, n_row_indices, n_col_indices, copy):
    mat = np.array(mat)

    n_rows, n_cols = mat.shape

    if n_row_indices == 0:
        row_indices = None
    else:
        row_indices = np.random.choice(np.arange(n_rows), size=min(n_rows, n_row_indices), replace=False)

    if n_col_indices == 0:
        col_indices = None
    else:
        col_indices = np.random.choice(np.arange(n_cols), size=min(n_cols, n_col_indices), replace=False)

    window = mat2d_window_from_indices(mat, row_indices, col_indices, copy)

    if row_indices is None:
        asserted_y_shape = n_rows
    else:
        asserted_y_shape = len(row_indices)
    assert window.shape[0] == asserted_y_shape

    if col_indices is None:
        asserted_x_shape = n_cols
    else:
        asserted_x_shape = len(col_indices)
    assert window.shape[1] == asserted_x_shape

    if row_indices is None:
        row_indices_check = np.arange(n_rows)
    else:
        row_indices_check = row_indices

    if col_indices is None:
        col_indices_check = np.arange(n_cols)
    else:
        col_indices_check = col_indices

    for w_y, m_y in enumerate(row_indices_check):
        for w_x, m_x in enumerate(col_indices_check):
            assert window[w_y, w_x] == mat[m_y, m_x]


@given(input=strategy_dtm_small(),
       logfn=st.sampled_from([np.log, np.log2, np.log10, np.log1p]),
       sparse_format=st.sampled_from(['csc', 'csr', 'lil', 'dok', 'coo']))
def test_partial_sparse_log(input, logfn, sparse_format):
    x = sparse.csc_matrix(input)
    if sparse_format != 'csc':
        x = x.asformat(sparse_format)

    y = partial_sparse_log(x, logfn=logfn)

    assert isinstance(y, sparse.spmatrix)
    assert y.getformat() == sparse_format
    assert y.shape == x.shape
    assert y.dtype.kind == 'f'

    if logfn is not np.log1p:
        y_dense = y.A
        n_gt_1 = np.sum(input > 1)
        assert y.nnz == n_gt_1
        if n_gt_1 > 0:
            assert np.all(y_dense[y_dense != 0.0] == logfn(input[input > 1]))
        else:
            assert np.sum(y_dense != 0.0) == n_gt_1


@given(dicts=st.lists(st.dictionaries(st.text(), st.integers())),
       sort_keys=st.booleans(),
       safe=st.booleans())
def test_merge_dicts(dicts, sort_keys, safe):
    all_keys = set()
    has_common_keys = False
    for d in dicts:
        ks = set(d.keys())
        if not has_common_keys and any(k in all_keys for k in ks):
            has_common_keys = True
        all_keys.update(ks)

    if len(dicts) > 1 and has_common_keys and safe:
        with pytest.raises(ValueError, match=r'^merging these containers would overwrite already existing contents'):
            merge_dicts(dicts, sort_keys=sort_keys, safe=safe)
    else:
        res = merge_dicts(dicts, sort_keys=sort_keys, safe=safe)
        assert isinstance(res, dict)
        n = sum(map(len, dicts))
        if has_common_keys:
            assert len(res) <= n
        else:
            assert len(res) == n
            for d in dicts:
                for k, v in d.items():
                    assert res[k] == v
        assert set(res.keys()) == all_keys
        if sort_keys:
            assert list(res.keys()) == sorted(all_keys)

@given(sets=st.lists(st.sets(st.integers())), safe=st.booleans())
def test_merge_sets(sets, safe):
    all_elems = set()
    has_common_elems = False
    for s in sets:
        if not has_common_elems and any(e in all_elems for e in s):
            has_common_elems = True
        all_elems.update(s)

    if len(sets) > 1 and has_common_elems and safe:
        with pytest.raises(ValueError, match=r'^merging these containers would overwrite already existing contents'):
            merge_sets(sets, safe=safe)
    else:
        res = merge_sets(sets, safe=safe)
        assert res == all_elems


@given(d=st.dictionaries(st.text(), st.integers()), n=st.integers())
def test_sample_dict(d, n):
    if 0 <= n <= len(d):
        res = sample_dict(d, n=n)
        assert isinstance(res, dict)
        assert len(res) == n
        assert set(res.keys()) <= set(d.keys())

        for k, v in res.items():
            assert v == d[k]
    else:
        with pytest.raises(ValueError):
            sample_dict(d, n=n)


@given(elems_dict=st.dictionaries(st.text(string.printable), st.floats(allow_nan=False, allow_infinity=False)),
       k=st.integers())
def test_greedy_partitioning(elems_dict, k):
    if k <= 0:
        with pytest.raises(ValueError):
            greedy_partitioning(elems_dict, k)
    else:
        bins = greedy_partitioning(elems_dict, k)

        if 1 < k <= len(elems_dict):
            assert k == len(bins)
        else:
            assert len(bins) == len(elems_dict)

        if k == 1:
            assert bins == elems_dict
        else:
            assert sum(len(b.keys()) for b in bins) == len(elems_dict)
            assert all((k in elems_dict.keys() for k in b.keys()) for b in bins)

            if k > len(elems_dict):
                assert all(len(b) == 1 for b in bins)


def test_combine_sparse_matrices_columnwise():
    m1 = sparse.coo_matrix(np.array([
        [1, 0, 3],
        [0, 2, 0],
    ]))
    
    cols1 = list('CAD')
    rows1 = [4, 0]   # row labels. can be integers!
    
    m2 = sparse.coo_matrix(np.array([
        [0, 0, 1, 2],
        [3, 4, 5, 6],
        [2, 1, 0, 0],
    ]))

    cols2 = list('DBCA')
    rows2 = [3, 1, 2]

    m3 = sparse.coo_matrix(np.array([
        [9, 8],
    ]))

    cols3 = list('BD')

    m4 = sparse.coo_matrix(np.array([
        [9],
        [8]
    ]))

    cols4 = list('A')

    m5 = sparse.coo_matrix((0, 0), dtype=int)

    cols5 = []

    expected_1_2 = np.array([
        [0, 0, 1, 3],
        [2, 0, 0, 0],
        [2, 0, 1, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
    ])

    expected_1_5 = np.array([
        [0, 0, 1, 3],
        [2, 0, 0, 0],
        [2, 0, 1, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
        [0, 9, 0, 8],   # 3
        [9, 0, 0, 0],   # 4
        [8, 0, 0, 0],   # 4
    ])

    expected_1_2_rows_sorted = np.array([
        [2, 0, 0, 0],
        [6, 4, 5, 3],
        [0, 1, 0, 2],
        [2, 0, 1, 0],
        [0, 0, 1, 3],
    ])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise([], [])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, ))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, list('X')))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m2, ), (cols1, cols2))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), [])

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, rows1))

    with pytest.raises(ValueError):
        combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, [0, 0, 0, 0]))

    # matrices 1 and 2, no row re-ordering
    res, res_cols = combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2))
    
    assert sparse.isspmatrix_csr(res)
    assert res.shape == (5, 4)
    assert np.all(res.A == expected_1_2)
    assert np.array_equal(res_cols, np.array(list('ABCD')))

    # matrices 1 and 2, re-order rows
    res, res_cols, res_rows = combine_sparse_matrices_columnwise((m1, m2), (cols1, cols2), (rows1, rows2))
    assert sparse.isspmatrix_csr(res)
    assert res.shape == (5, 4)
    assert np.all(res.A == expected_1_2_rows_sorted)
    assert np.array_equal(res_cols, np.array(list('ABCD')))
    assert np.array_equal(res_rows, np.arange(5))

    # matrices 1 to 5, no row re-ordering
    res, res_cols = combine_sparse_matrices_columnwise((m1, m2, m3, m4, m5), (cols1, cols2, cols3, cols4, cols5))

    assert sparse.isspmatrix_csr(res)
    assert np.all(res.A == expected_1_5)
    assert np.array_equal(res_cols, np.array(list('ABCD')))


@pytest.mark.parametrize('testfn, testargs, expargs1, expargs2', [
    (lambda x, y: ..., {'x': 1, 'y': 2, 'z': 3}, {'x': 1, 'y': 2}, {'z': 3}),
    (lambda: ..., {'x': 1, 'y': 2, 'z': 3}, {}, {'x': 1, 'y': 2, 'z': 3}),
    (lambda x, y, z: ..., {'x': 1, 'y': 2, 'z': 3}, {'x': 1, 'y': 2, 'z': 3}, {}),
])
def test_split_func_args(testfn, testargs, expargs1, expargs2):
    res = split_func_args(testfn, testargs)
    assert isinstance(res, tuple) and len(res) == 2
    args1, args2 = res
    assert args1 == expargs1
    assert args2 == expargs2


@given(cs=st.lists(st.integers(-1, 5), min_size=0, max_size=3),
       container=st.sampled_from([list, tuple]),
       dtype=st.sampled_from([int, float]))
def test_check_context_size(cs, container, dtype):
    cs = container(map(dtype, cs))
    if len(cs) == 1:
        cs = cs[0]

    if isinstance(cs, float):
        with pytest.raises(ValueError, match='`context_size` must be integer or list/tuple'):
            check_context_size(cs)
    elif isinstance(cs, (tuple, list)):
        if len(cs) != 2:
            with pytest.raises(ValueError, match='`context_size` must be list/tuple of length 2'):
                check_context_size(cs)
        elif dtype is float or min(cs) < 0 or all(s == 0 for s in cs):
            with pytest.raises(ValueError, match='^`context_size` must contain'):
                check_context_size(cs)
        else:
            res = check_context_size(cs)
            assert res == tuple(cs)
    else:
        if dtype is float or cs <= 0:
            with pytest.raises(ValueError, match='^`context_size` must contain'):
                check_context_size(cs)
        else:
            res = check_context_size(cs)
            assert res == (cs, cs)


@given(m=strategy_2d_array(float, -10, 10),
       to_int=st.booleans(),
       sparsetype=st.sampled_from(['csc', 'csr', 'coo']),
       pass_dimnames=st.sampled_from(['none', 'rows', 'cols', 'both']),
       return_dimnames=st.booleans())
def test_sparsemat_rinterop(m, to_int, sparsetype, pass_dimnames, return_dimnames):
    try:
        from tmtoolkit.utils import sparsemat_from_r, sparsemat_to_r, RS4
    except ImportError:
        pytest.skip('packages for R interoperability not installed')
        return

    if to_int:
        m = m.astype('int')

    spmatfn = getattr(sparse, f'{sparsetype}_matrix')

    s = spmatfn(m)

    rownames = None
    colnames = None
    if pass_dimnames in {'rows', 'both'}:
        rownames = [f'r{i}' for i in range(m.shape[0])]
    if pass_dimnames in {'cols', 'both'}:
        colnames = [f'r{i}' for i in range(m.shape[1])]

    rs = sparsemat_to_r(s, rownames=rownames, colnames=colnames)
    assert isinstance(rs, RS4)
    s_ = sparsemat_from_r(rs, return_dimnames=return_dimnames)

    if return_dimnames:
        assert isinstance(s_, tuple)
        assert len(s_) == 3
        s_, returned_rows, returned_cols = s_
        assert returned_rows == rownames
        assert returned_cols == colnames

    assert isinstance(s_, sparse.csc_matrix)
    assert str(s_.dtype).startswith('float')   # can't recover type, is always float

    if to_int:
        m_ = s_.todense().astype('int')
    else:
        m_ = s_.todense()

    assert np.allclose(m, m_)


@given(m=strategy_2d_array(float, -10, 10),
       to_int=st.booleans(),
       pass_dimnames=st.sampled_from(['none', 'rows', 'cols', 'both']),
       return_dimnames=st.booleans())
def test_mat_rinterop(m, to_int, pass_dimnames, return_dimnames):
    try:
        from tmtoolkit.utils import mat_from_r, mat_to_r, robjects
    except ImportError:
        pytest.skip('packages for R interoperability not installed')
        return

    if to_int:
        m = m.astype('int')

    rownames = None
    colnames = None
    if pass_dimnames in {'rows', 'both'}:
        rownames = [f'r{i}' for i in range(m.shape[0])]
    if pass_dimnames in {'cols', 'both'}:
        colnames = [f'r{i}' for i in range(m.shape[1])]

    rm = mat_to_r(m, rownames=rownames, colnames=colnames)
    m_ = mat_from_r(rm, return_dimnames=return_dimnames)

    if return_dimnames:
        assert isinstance(m_, tuple)
        assert len(m_) == 3
        m_, returned_rows, returned_cols = m_
        assert returned_rows == rownames
        assert returned_cols == colnames

    assert isinstance(m_, np.ndarray)

    if to_int:
        assert isinstance(rm, robjects.vectors.IntMatrix)
        assert np.issubdtype(m_.dtype, 'int32') or np.issubdtype(m_.dtype, 'int64')
    else:
        assert isinstance(rm, robjects.vectors.FloatMatrix)
        assert np.issubdtype(m_.dtype, 'float32') or np.issubdtype(m_.dtype, 'float64')

    assert np.allclose(m, m_)
