"""
Tests for tmtoolkit.tokenseq module.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import string
from collections import Counter

import numpy as np
import pytest
import random
from hypothesis import given, strategies as st
from scipy import sparse

from tmtoolkit import tokenseq
from tmtoolkit.utils import as_chararray, flatten_list

from ._testtools import strategy_tokens, strategy_2d_array, strategy_lists_of_tokens, identity


@given(s=strategy_tokens(string.printable),
       s_type=st.sampled_from(['list', 'tuple', 'nparray']),
       left=st.integers(-1, 5),
       right=st.integers(-1, 5),
       left_symbol=st.sampled_from(['', 'X', 'START']),
       right_symbol=st.sampled_from(['', 'Y', 'END']),
       skip_empty=st.booleans())
def test_pad_sequence_str(s, s_type, left, right, left_symbol, right_symbol, skip_empty):
    _test_pad_sequence(s, s_type, 'str', left, right, left_symbol, right_symbol, skip_empty)


@given(s=st.lists(st.integers(21, 40)),
       s_type=st.sampled_from(['list', 'tuple', 'nparray']),
       left=st.integers(-1, 5),
       right=st.integers(-1, 5),
       left_symbol=st.sampled_from([0, -1, 10]),
       right_symbol=st.sampled_from([0, -2, 20]),
       skip_empty=st.booleans())
def test_pad_sequence_int(s, s_type, left, right, left_symbol, right_symbol, skip_empty):
    _test_pad_sequence(s, s_type, 'int', left, right, left_symbol, right_symbol, skip_empty)


def _test_pad_sequence(s, s_type, el_type, left, right, left_symbol, right_symbol, skip_empty):
    if s_type == 'tuple':
        s = tuple(s)
        check_type = tuple
    elif s_type == 'nparray':
        if el_type == 'str':
            s = as_chararray(s)
        else:
            s = np.array(s, dtype='int')
        check_type = np.ndarray
    else:
        check_type = list

    args = dict(left=left, right=right, left_symbol=left_symbol, right_symbol=right_symbol, skip_empty=skip_empty)

    if left < 0 or right < 0:
        with pytest.raises(ValueError):
            tokenseq.pad_sequence(s, **args)
    else:
        spad = tokenseq.pad_sequence(s, **args)
        assert isinstance(spad, check_type)

        if s_type == 'nparray':
            assert np.issubdtype(spad.dtype, el_type)

            if el_type == 'int':
                el_type_check = np.int64
            else:
                el_type_check = str
        else:
            if el_type == 'int':
                el_type_check = int
            else:
                el_type_check = str

        assert all(isinstance(t, el_type_check) for t in list(spad))

        assert len(spad) >= len(s)

        if (skip_empty and len(s) == 0) or left == right == 0:
            assert list(spad) == list(s)
        else:
            assert len(spad) == len(s) + left + right
            assert list(spad[:left]) == [left_symbol] * left
            if right > 0:
                assert list(spad[-right:]) == [right_symbol] * right
                assert list(spad[left:-right]) == list(s)
            else:
                assert list(spad[left:]) == list(s)


@pytest.mark.parametrize('tokens, expected', [
    ([], []),
    ([''], [0]),
    (['a'], [1]),
    (['abc'], [3]),
    (['abc', 'd'], [3, 1]),
])
def test_token_lengths(tokens, expected):
    assert tokenseq.token_lengths(tokens) == expected


@given(tokens=strategy_tokens(string.printable),
       as_array=st.booleans())
def test_token_lengths_hypothesis(tokens, as_array):
    if as_array:
        tokens = as_chararray(tokens)

    res = tokenseq.token_lengths(tokens)

    assert isinstance(res, list)
    assert len(res) == len(tokens)
    assert all([isinstance(n, int) and n >= 0 for n in res])


@given(tokens=strategy_tokens())
def test_unique_chars_hypothesis(tokens):
    res = tokenseq.unique_chars(tokens)
    assert isinstance(res, set)
    assert all(isinstance(c, str) for c in res)
    assert len(res) <= sum(map(len, tokens))

    for t in tokens:
        for c in t:
            assert c in res


@given(tokens=strategy_tokens(string.printable),
       tokens_as_array=st.booleans(),
       collapse=st.one_of(st.text(), strategy_tokens(string.printable)),
       collapse_as_array=st.booleans())
def test_collapse_tokens(tokens, tokens_as_array, collapse, collapse_as_array):
    def _common_result_check(res):
        assert isinstance(res, str)
        for t in tokens:
            assert t in res

    if tokens_as_array:
        tokens = as_chararray(tokens)
    if collapse_as_array and not isinstance(collapse, str):
        collapse = as_chararray(collapse)

    if isinstance(collapse, str):
        res = tokenseq.collapse_tokens(tokens, collapse=collapse)
        _common_result_check(res)

        if collapse:
            assert res.count(collapse) >= len(tokens) - 1
    else:
        if len(tokens) == len(collapse):
            res = tokenseq.collapse_tokens(tokens, collapse=collapse)
            _common_result_check(res)

            for t in collapse:
                assert t in res
        else:
            with pytest.raises(ValueError, match='if `collapse` is given as sequence, it must have the same length as '
                                                 '`tokens`'):
                tokenseq.collapse_tokens(tokens, collapse=collapse)


@given(tokens=strategy_tokens(string.ascii_letters),
       tokens_as_hashes=st.booleans(),
       tokens_as_array=st.booleans(),
       special_tokens=st.one_of(st.none(), strategy_tokens(string.ascii_letters, min_size=1)),
       collapse=st.booleans())
def test_token_hash_convert(tokens, tokens_as_hashes, tokens_as_array, special_tokens, collapse):
    # manually make bidicts
    stringstore = {t: hash(t) for t in tokens}
    stringstore.update({h: t for t, h in stringstore.items()})

    if special_tokens is not None:
        special_tokens_dict = dict(enumerate(special_tokens))
        special_tokens_dict.update({t: i for i, t in special_tokens_dict.items()})
    else:
        special_tokens_dict = None

    if tokens_as_hashes:
        tokens = list(map(hash, tokens))

    if tokens_as_array:
        tokens = np.array(tokens, dtype='int64' if tokens_as_hashes else 'str')

    collapse = ' ' if collapse and tokens_as_hashes else None
    res = tokenseq.token_hash_convert(tokens, stringstore=stringstore, special_tokens=special_tokens_dict,
                                                collapse=collapse, arr_dtype_for_hashes='int64')

    if collapse == ' ':
        assert isinstance(res, str)
        res = res.split(collapse)

        if special_tokens_dict:
            assert all(t in stringstore or t in special_tokens_dict for t in res if t != '')
        else:
            assert all(t in stringstore for t in res if t != '')

        assert all(isinstance(t, str) for t in res)
    else:
        if tokens_as_array:
            assert isinstance(res, np.ndarray)
        else:
            assert isinstance(res, list)

        assert len(res) == len(tokens)

        if special_tokens_dict:
            assert all(t in stringstore or t in special_tokens_dict for t in res)
        else:
            assert all(t in stringstore for t in res)

        if tokens_as_hashes:
            if tokens_as_array:
                assert np.issubdtype(res.dtype, 'str')
            else:
                assert all(isinstance(t, str) for t in res)


@given(xy=strategy_2d_array(int, 0, 100, min_side=2, max_side=100),
       as_prob=st.booleans(),
       n_total_factor=st.floats(min_value=1, max_value=10, allow_nan=False),
       k=st.integers(min_value=0, max_value=5),
       normalize=st.booleans())
def test_pmi_vectors_hypothesis(xy, as_prob, n_total_factor, k, normalize):
    size = len(xy)
    xy = xy[:, 0:2]
    x = xy[:, 0]
    y = xy[:, 1]
    xy = np.min(xy, axis=1) * np.random.uniform(0, 1, size)
    n_total = 1 + n_total_factor * (np.sum(x) + np.sum(y))

    if as_prob:
        x = x / n_total
        y = y / n_total
        xy = xy / n_total
        n_total = None

    if k < 1 or (k > 1 and normalize):
        with pytest.raises(ValueError):
            tokenseq.pmi(x, y, xy, n_total=n_total, k=k, normalize=normalize)
    else:
        with pytest.raises(ValueError):
            tokenseq.pmi(x, y, xy, n_total=n_total, k=k, alpha=0.75, normalize=normalize)

        res = tokenseq.pmi(x, y, xy, n_total=n_total, k=k, normalize=normalize)
        assert isinstance(res, np.ndarray)
        assert len(res) == len(x)

        if np.all(x > 0) and np.all(y > 0):
            assert np.sum(np.isnan(res)) == 0
            if normalize:
                assert np.all(res == tokenseq.npmi(x, y, xy, n_total=n_total))
                assert np.all(res >= -1) and np.all(res <= 1)
            elif k == 2:
                assert np.all(res == tokenseq.pmi2(x, y, xy, n_total=n_total))
            elif k == 3:
                assert np.all(res == tokenseq.pmi3(x, y, xy, n_total=n_total))


@given(xy=strategy_2d_array(int, 0, 100, min_side=2, max_side=100),
       as_prob=st.booleans(),
       as_sparse=st.sampled_from([None, 'csc', 'csr']),
       k=st.integers(min_value=0, max_value=5),
       alpha=st.one_of(st.just(1.0), st.floats(min_value=0.1, max_value=2.0)),
       normalize=st.booleans())
def test_pmi_matrix_hypothesis(xy, as_prob, as_sparse, k, alpha, normalize):
    if as_prob:
        if np.sum(xy) > 0:
            xy = xy / np.sum(xy)
        else:
            xy = xy.astype('float')

    if as_sparse:
        xy = sparse.coo_matrix(xy).asformat(as_sparse)

    kwargs = dict(k=k, normalize=normalize)

    if k < 1 or (k > 1 and normalize):
        with pytest.raises(ValueError):
            tokenseq.pmi(xy, **kwargs)
    else:
        with pytest.raises(ValueError):
            tokenseq.pmi(xy.A[np.newaxis] if as_sparse else xy[np.newaxis], **kwargs)
        with pytest.raises(ValueError):
            tokenseq.pmi(xy, xy[0], **kwargs)
        with pytest.raises(ValueError):
            tokenseq.pmi(xy, xy[0], xy[1], **kwargs)

        if not as_sparse:
            if as_prob:
                with pytest.raises(ValueError):
                    tokenseq.pmi(xy - 10.0, **kwargs)
                with pytest.raises(ValueError):
                    tokenseq.pmi(xy + 10.0, **kwargs)
                if alpha != 1.0:
                    with pytest.raises(ValueError):
                        tokenseq.pmi(xy, **kwargs, alpha=alpha)
            else:
                with pytest.raises(ValueError):
                    tokenseq.pmi(xy - 101, **kwargs)

        if as_prob and not np.isclose(np.sum(xy), 1.0):
            with pytest.raises(ValueError):
                tokenseq.pmi(xy, **kwargs)
        elif not as_prob and np.sum(xy) == 0:
            with pytest.raises(ValueError):
                tokenseq.pmi(xy, **kwargs)
        else:
            res = tokenseq.pmi(xy, **kwargs, alpha=1.0 if as_prob else alpha)
            assert isinstance(res, np.ndarray)
            assert res.shape == xy.shape

            if alpha == 1.0:
                if normalize:
                    assert np.array_equal(res, tokenseq.npmi(xy), equal_nan=True)
                    if np.sum(np.isnan(res)) == 0:
                        assert np.all(res >= -1) and np.all(res <= 1)
                elif k == 2:
                    assert np.array_equal(res, tokenseq.pmi2(xy), equal_nan=True)
                elif k == 3:
                    assert np.array_equal(res, tokenseq.pmi3(xy), equal_nan=True)


@given(xy=strategy_2d_array(int, 0, 100, min_side=2, max_side=100),
       as_prob=st.booleans(),
       n_total_factor=st.floats(min_value=1, max_value=10, allow_nan=False))
def test_ppmi_hypothesis(xy, as_prob, n_total_factor):
    size = len(xy)
    xy = xy[:, 0:2]
    x = xy[:, 0]
    y = xy[:, 1]
    xy = np.min(xy, axis=1) * np.random.uniform(0, 1, size)
    n_total = 1 + n_total_factor * (np.sum(x) + np.sum(y))

    if as_prob:
        x = x / n_total
        y = y / n_total
        xy = xy / n_total
        n_total = None

    with pytest.raises(ValueError):
        tokenseq.ppmi(x, y, xy, n_total=n_total, alpha=0.75)
    with pytest.raises(ValueError):
        tokenseq.ppmi(x, y, xy, n_total=n_total, add_k_smoothing=1.5)

    res = tokenseq.ppmi(x, y, xy, n_total=n_total)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(x)

    if np.all(x > 0) and np.all(y > 0):
        assert np.sum(np.isnan(res)) == 0
        assert np.all(res >= 0)


@given(xy=strategy_2d_array(int, 0, 100, min_side=2, max_side=100),
       as_prob=st.booleans(),
       as_sparse=st.sampled_from([None, 'csc', 'csr']),
       alpha=st.one_of(st.just(1.0), st.floats(min_value=0.1, max_value=2.0)),
       add_k_smoothing=st.one_of(st.just(0.0), st.floats(min_value=0.1, max_value=2.0)))
def test_ppmi_matrix_hypothesis(xy, as_prob, as_sparse, alpha, add_k_smoothing):
    if as_prob:
        if np.sum(xy) > 0:
            xy = xy / np.sum(xy)
        else:
            xy = xy.astype('float')

    xy_dense = None
    if as_sparse:
        xy_dense = xy
        xy = sparse.coo_matrix(xy).asformat(as_sparse)

    kwargs = dict(add_k_smoothing=add_k_smoothing, alpha=alpha)

    if as_prob and add_k_smoothing != 0.0:
        with pytest.raises(ValueError):
            tokenseq.ppmi(xy, **kwargs)
    else:
        if as_prob and not np.isclose(np.sum(xy), 1.0):
            with pytest.raises(ValueError):
                tokenseq.ppmi(xy, **kwargs)
        elif as_prob and (alpha != 1.0 or add_k_smoothing != 0.0):
            with pytest.raises(ValueError):
                tokenseq.ppmi(xy, **kwargs)
        elif not as_prob and np.sum(xy) == 0 and add_k_smoothing == 0.0:
            with pytest.raises(ValueError):
                tokenseq.ppmi(xy, **kwargs)
        else:
            res = tokenseq.ppmi(xy, **kwargs)
            if as_sparse and add_k_smoothing == 0.0:
                assert isinstance(res, sparse.spmatrix)
                res_dense = res.A
            else:
                assert isinstance(res, np.ndarray)
                res_dense = res
            assert res.shape == xy.shape

            if np.sum(np.isnan(res_dense)) == 0:
                assert np.all(res_dense >= 0)

            if as_sparse and add_k_smoothing == 0.0:
                res_dense2 = tokenseq.ppmi(xy_dense, **kwargs)
                if np.sum(np.isnan(res_dense2)) == 0:
                    assert np.allclose(res_dense, res_dense2)


@given(sentences=strategy_lists_of_tokens(string.printable),
       min_count=st.integers(),
       pass_embed_tokens=st.integers(min_value=0, max_value=3),
       tokens_as_hashes=st.booleans(),
       return_vocab=st.booleans(),
       return_bigrams_with_indices=st.booleans()
       )
def test_token_collocation_matrix_hypothesis(sentences, min_count, pass_embed_tokens, tokens_as_hashes,
                                             return_vocab, return_bigrams_with_indices):
    if tokens_as_hashes:
        sentences = [list(map(hash, sent)) for sent in sentences]
    tok = flatten_list(sentences)

    if pass_embed_tokens > 0:
        embed_tokens = random.choices(tok, k=min(pass_embed_tokens, len(tok)))
    else:
        embed_tokens = None

    args = dict(sentences=sentences, min_count=min_count, embed_tokens=embed_tokens, tokens_as_hashes=tokens_as_hashes,
                return_vocab=return_vocab, return_bigrams_with_indices=return_bigrams_with_indices)

    if min_count < 0:
        with pytest.raises(ValueError):
            tokenseq.token_collocation_matrix(**args)
    else:
        res = tokenseq.token_collocation_matrix(**args)
        vocab1 = vocab2 = bigrams_w_indices = None

        if return_vocab:
            assert isinstance(res, tuple)
            assert len(res) == 3
            mat, vocab1, vocab2 = res
        elif return_bigrams_with_indices:
            assert isinstance(res, tuple)
            assert len(res) == 2
            mat, bigrams_w_indices = res
        else:
            mat = res

        assert isinstance(mat, sparse.csr_matrix)
        assert np.issubdtype(mat.dtype, 'uint32')

        if len(tok) < 2:
            assert mat.nnz == 0
        elif not embed_tokens and min_count == 0:
            assert mat.nnz > 0

        assert mat.shape[0] > 0
        assert mat.shape[1] > 0

        if mat.nnz > 0 and min_count == 0:
            if return_vocab:
                assert mat.shape == (len(vocab1), len(vocab2))

            if return_bigrams_with_indices:
                assert len(bigrams_w_indices) == 2
                bigrams, bigram_ind = bigrams_w_indices
                assert len(bigrams) == len(bigram_ind)
                assert np.prod(mat.shape) == len(bigrams)


@pytest.mark.parametrize('args, expected', [
    (
        {},
        [(('e', 'f'), 1.6094379124341005),
         (('d', 'e'), 1.6094379124341),
         (('b', 'c'), 1.3217558399823195),
         (('a', 'b'), 1.09861228866811),
         (('f', 'b'), 1.09861228866811),
         (('c', 'd'), 1.0986122886681096),
         (('c', 'e'), 0.6931471805599454),
         (('c', 'b'), 0.18232155679395445),
         (('e', 'b'), 0.0)]
    ),
    (
        dict(min_count=2),
        [(('e', 'f'), 1.6094379124341003),
         (('b', 'c'), 0.916290731874155),
         (('c', 'b'), 0.9162907318741549),
         (('c', 'e'), 0.9162907318741549)]
    ),
    (
        dict(threshold=0.5),
        [(('e', 'f'), 1.6094379124341005),
         (('d', 'e'), 1.6094379124341),
         (('b', 'c'), 1.3217558399823195),
         (('a', 'b'), 1.09861228866811),
         (('f', 'b'), 1.09861228866811),
         (('c', 'd'), 1.0986122886681096),
         (('c', 'e'), 0.6931471805599454)]
    ),
    (
        dict(min_count=2, threshold=0.5, glue='_&_'),
        [('e_&_f', 1.6094379124341003),
         ('b_&_c', 0.916290731874155),
         ('c_&_b', 0.9162907318741549),
         ('c_&_e', 0.9162907318741549)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.npmi),
        [(('b', 'c'), 1.0),
         (('e', 'f'), 1.0),
         (('c', 'b'), 0.5693234419266069),
         (('c', 'e'), 0.5693234419266069)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.pmi2),
        [(('b', 'c'), 0.0),
         (('e', 'f'), 0.0),
         (('c', 'b'), -0.6931471805599454),
         (('c', 'e'), -0.6931471805599454)]
    ),
    (
        dict(min_count=2, statistic=tokenseq.pmi3),
        [(('b', 'c'), -0.916290731874155),
         (('e', 'f'), -1.6094379124341003),
         (('c', 'b'), -2.302585092994046),
         (('c', 'e'), -2.302585092994046)]
    )
])
def test_token_collocations(args, expected):
    sentences = ['a b c d e f b c b'.split(),
                 'c e b c b c e f'.split()]
    res = tokenseq.token_collocations(sentences, **args)
    colloc, stat = zip(*res)
    expected_colloc, expected_stat = zip(*expected)
    assert colloc == expected_colloc
    assert np.allclose(stat, expected_stat)


@given(sentences=strategy_lists_of_tokens(string.printable),
       threshold=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
       min_count=st.integers(),
       pass_embed_tokens=st.integers(min_value=0, max_value=3),
       statistic=st.sampled_from([tokenseq.pmi, tokenseq.npmi, tokenseq.pmi2, tokenseq.pmi3,
                                  tokenseq.ppmi, identity]),
       glue=st.one_of(st.none(), st.text(string.printable)),
       return_statistic=st.booleans(),
       rank=st.sampled_from([None, 'asc', 'desc'])
       )
def test_token_collocations_hypothesis(sentences, threshold, min_count, pass_embed_tokens, statistic,
                                       glue, return_statistic, rank):
    ngramsize = 2
    tok = flatten_list(sentences)

    if pass_embed_tokens > 0:
        embed_tokens = random.choices(tok, k=min(pass_embed_tokens, len(tok)))
    else:
        embed_tokens = None

    args = dict(sentences=sentences, threshold=threshold, min_count=min_count, embed_tokens=embed_tokens,
                statistic=statistic, glue=glue, return_statistic=return_statistic, rank=rank)

    if min_count < 0:
        with pytest.raises(ValueError):
            tokenseq.token_collocations(**args)
    else:
        res = tokenseq.token_collocations(**args)
        assert isinstance(res, list)
        assert len(res) <= max(1, len(tok) - ngramsize + 1)

        statvalues = []
        for row in res:
            if return_statistic:
                assert isinstance(row, tuple)
                assert len(row) == 2
                colloc, stat = row
                assert isinstance(stat, float)
                if threshold:
                    assert stat >= threshold
                if statistic is identity:
                    assert stat >= min_count
                if rank:
                    statvalues.append(stat)
            else:
                colloc = row

            if glue is None:
                assert isinstance(colloc, tuple)
                assert all([isinstance(t, str) for t in colloc])
                if embed_tokens:
                    assert len(colloc) >= ngramsize
                else:
                    assert len(colloc) == ngramsize
            else:
                assert isinstance(colloc, str)
                assert glue in colloc
        if rank:
            assert statvalues == sorted(statvalues, reverse=rank == 'desc')


@pytest.mark.parametrize('pattern, tokens, match_type, ignore_case, glob_method, expected', [
    ('a', [], 'exact', False, 'match', []),
    ('', [], 'exact', False, 'match', []),
    ('', ['a', ''], 'exact', False, 'match', [False, True]),
    ('a', ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    ('a', np.array(['a', 'b', 'c']), 'exact', False, 'match', [True, False, False]),
    ('A', ['a', 'b', 'c'], 'exact', False, 'match', [False, False, False]),
    ('A', ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    (r'foo$', ['a', 'bfoo', 'c'], 'regex', False, 'match', [False, True, False]),
    (r'foo$', ['a', 'bFOO', 'c'], 'regex', False, 'match', [False, False, False]),
    (r'foo$', ['a', 'bFOO', 'c'], 'regex', True, 'match', [False, True, False]),
    (r'foo*', ['a', 'food', 'c'], 'glob', False, 'match', [False, True, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', False, 'match', [False, False, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', True, 'match', [False, True, False]),
    (r'foo*', ['a', 'FOOd', 'c'], 'glob', True, 'search', [False, True, False]),
])
def test_token_match(pattern, tokens, match_type, ignore_case, glob_method, expected):
    assert np.array_equal(tokenseq.token_match(pattern, tokens, match_type, ignore_case, glob_method),
                          np.array(expected))


@pytest.mark.parametrize('pattern, tokens, match_type, ignore_case, glob_method, expected', [
    ('a', [], 'exact', False, 'match', []),
    ('', [], 'exact', False, 'match', []),
    ('', ['a', ''], 'exact', False, 'match', [False, True]),
    ('a', ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    (['a'], ['a', 'b', 'c'], 'exact', False, 'match', [True, False, False]),
    (['a', 'c'], ['a', 'b', 'c'], 'exact', False, 'match', [True, False, True]),
    (('a', 'c'), np.array(['a', 'b', 'c']), 'exact', False, 'match', [True, False, True]),
    ({'A'}, ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    ({'A', 'a'}, ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False]),
    (['A', 'A'], ['a', 'b', 'c'], 'exact', True, 'match', [True, False, False])
])
def test_token_match_multi_pattern(pattern, tokens, match_type, ignore_case, glob_method, expected):
    assert np.array_equal(
        tokenseq.token_match_multi_pattern(pattern, tokens, match_type, ignore_case, glob_method),
        np.array(expected))


def test_token_match_subsequent():
    tok = ['green', 'test', 'emob', 'test', 'greener', 'tests', 'test', 'test']

    with pytest.raises(ValueError):
        tokenseq.token_match_subsequent('pattern', tok)

    with pytest.raises(ValueError):
        tokenseq.token_match_subsequent(['pattern'], tok)

    assert tokenseq.token_match_subsequent(['a', 'b'], []) == []

    assert tokenseq.token_match_subsequent(['foo', 'bar'], tok) == []

    res = tokenseq.token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([0, 1]))
    assert np.array_equal(res[1], np.array([4, 5]))

    res = tokenseq.token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([0, 1, 2]))
    assert np.array_equal(res[1], np.array([4, 5, 6]))


@given(tokens=st.lists(st.text()), n_patterns=st.integers(0, 4))
def test_token_match_subsequent_hypothesis(tokens, n_patterns):
    tokens = np.array(tokens)

    n_patterns = min(len(tokens), n_patterns)

    pat_ind = np.arange(n_patterns)
    np.random.shuffle(pat_ind)
    patterns = list(tokens[pat_ind])

    if len(patterns) < 2:
        with pytest.raises(ValueError):
            tokenseq.token_match_subsequent(patterns, tokens)
    else:
        res = tokenseq.token_match_subsequent(patterns, tokens)

        assert isinstance(res, list)
        if len(tokens) == 0:
            assert res == []
        else:
            for ind in res:
                assert len(ind) == len(patterns)
                assert np.all(ind >= 0)
                assert np.all(ind < len(tokens))
                assert np.all(np.diff(ind) == 1)   # subsequent words
                assert np.array_equal(tokens[ind], patterns)


def test_token_glue_subsequent():
    tok = ['green', 'test', 'emob', 'test', 'greener', 'tests', 'test', 'test']

    with pytest.raises(ValueError):
        tokenseq.token_join_subsequent(tok, 'invalid')

    assert tokenseq.token_join_subsequent(tok, []) == tok

    matches = tokenseq.token_match_subsequent(['green*', 'test*'], tok, match_type='glob')
    assert tokenseq.token_join_subsequent(tok, matches) == ['green_test', 'emob', 'test', 'greener_tests', 'test',
                                                            'test']

    matches = tokenseq.token_match_subsequent(['green*', 'test*', '*'], tok, match_type='glob')
    assert tokenseq.token_join_subsequent(tok, matches) == ['green_test_emob', 'test', 'greener_tests_test', 'test']


@given(tokens=st.lists(st.text(string.printable)), n_patterns=st.integers(0, 4))
def test_token_glue_subsequent_hypothesis(tokens, n_patterns):
    tokens_arr = np.array(tokens)

    n_patterns = min(len(tokens), n_patterns)

    pat_ind = np.arange(n_patterns)
    np.random.shuffle(pat_ind)
    patterns = list(tokens_arr[pat_ind])

    if len(patterns) > 1:
        matches = tokenseq.token_match_subsequent(patterns, tokens)
        assert tokenseq.token_join_subsequent(tokens, []) == tokens

        if len(tokens) == 0:
            assert tokenseq.token_join_subsequent(tokens, matches) == []
        elif len(matches) == 0:
            assert tokenseq.token_join_subsequent(tokens, matches) == tokens
        else:
            res = tokenseq.token_join_subsequent(tokens, matches)
            assert isinstance(res, list)
            assert 0 < len(res) < len(tokens)

            for ind in matches:
                assert '_'.join(tokens_arr[ind]) in res


@given(tokens=st.lists(st.text(string.printable)),
       n=st.integers(-1, 5),
       join=st.booleans(),
       join_str=st.text(string.printable, max_size=3),
       ngram_container=st.sampled_from([list, tuple]),
       pass_embed_tokens=st.integers(min_value=0, max_value=3),
       keep_embed_tokens=st.booleans())
def test_token_ngrams_hypothesis(tokens, n, join, join_str, ngram_container, pass_embed_tokens, keep_embed_tokens):
    if pass_embed_tokens:
        embed_tokens = set(random.choices(tokens, k=min(pass_embed_tokens, len(tokens))))
    else:
        embed_tokens = None

    args = dict(n=n, join=join, join_str=join_str, ngram_container=ngram_container,
                embed_tokens=embed_tokens, keep_embed_tokens=keep_embed_tokens)

    if n < 1:
        with pytest.raises(ValueError):
            tokenseq.token_ngrams(tokens, **args)
    else:
        res = tokenseq.token_ngrams(tokens, **args)
        assert isinstance(res, list)

        n_tok = len(tokens)

        if n_tok < n:
            if n_tok == 0:
                assert res == []
            else:
                assert len(res) == 1
                if join:
                    assert res == [join_str.join(tokens)]
                else:
                    assert res == [ngram_container(tokens)]
        else:
            if not pass_embed_tokens or keep_embed_tokens:
                assert len(res) == n_tok - n + 1

            if join:
                assert all([isinstance(g, str) for g in res])
                if n > 1:
                    assert all([join_str in g for g in res])
            else:
                assert all([isinstance(g, ngram_container) for g in res])

                if embed_tokens:
                    if keep_embed_tokens:
                        assert all([len(g) >= n for g in res])
                        assert all([any(t in g for t in embed_tokens) for g in res if len(g) > n])
                    else:
                        assert all([len(g) == n for g in res])
                        assert all([t not in embed_tokens for g in res for t in g])
                else:
                    assert all([len(g) == n for g in res])
                    tokens_ = list(res[0])
                    if len(res) > 1:
                        for g in res[1:]:
                            tokens_.extend(g[n-1:])

                    assert tokens_ == tokens
