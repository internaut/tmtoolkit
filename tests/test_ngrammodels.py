import math
import random
from collections import Counter

import pytest
from hypothesis import given, strategies as st

from tmtoolkit.corpus import Corpus, vocabulary
from tmtoolkit.ngrammodels import NGramModel, SPECIAL_TOKENS, SENT_END, SENT_START, OOV
from ._testtextdata import textdata_sm


@pytest.fixture(scope='module')
def small_corpus():
    return Corpus({'d1': 'I am Sam', 'd2': 'Sam I am', 'd3': 'I do not like green eggs and ham'}, language='en')


@pytest.fixture(scope='module')
def textdata_en():
    return textdata_sm['en']


@pytest.fixture(scope='module')
def corpus_en(textdata_en):
    return Corpus(textdata_en, language='en')


@given(n=st.integers(-1, 10),
       add_k_smoothing=st.floats(-1.0, 2.0),
       keep_vocab=st.one_of(st.none(), st.integers(-1, 100), st.floats(-1.0, 2.0)),
       tokens_as_hashes=st.booleans())
def test_init_and_properties(n, add_k_smoothing, keep_vocab, tokens_as_hashes):
    opts = dict(n=n, add_k_smoothing=add_k_smoothing, keep_vocab=keep_vocab, tokens_as_hashes=tokens_as_hashes)

    if n <= 0 or add_k_smoothing < 0 or \
            (keep_vocab is not None and (keep_vocab <= 0 or (isinstance(keep_vocab, float) and keep_vocab > 1.0))):
        with pytest.raises(ValueError):
            NGramModel(**opts)
    else:
        ng = NGramModel(**opts)
        assert ng.n == n
        assert ng.k == add_k_smoothing
        assert ng.keep_vocab == keep_vocab
        assert ng.tokens_as_hashes == tokens_as_hashes
        assert ng.vocab_size_ == 0
        assert ng.n_unigrams_ == 0
        assert isinstance(ng.ngram_counts_, Counter)
        assert len(ng.ngram_counts_) == 0
        assert ng.stringstore is None


def test_with_small_corpus_bigrams(small_corpus):
    ngmodel = NGramModel(2, add_k_smoothing=0, tokens_as_hashes=False)
    ngmodel.fit(small_corpus)

    assert math.isclose(ngmodel.prob('I', log=False), 3/20)
    assert math.isclose(ngmodel.prob('Sam', 'am', log=False), 1/2)
    assert math.isclose(ngmodel.prob('Sam', '<s>', log=False), 1/3)
    assert math.isclose(ngmodel.prob('I', '<s>', log=False), 2/3)
    assert math.isclose(ngmodel.prob('</s>', 'Sam', log=False), 1/2)
    assert math.isclose(ngmodel.prob('am', 'I', log=False), 2/3)
    assert math.isclose(ngmodel.prob('do', 'I', log=False), 1/3)
    assert math.isclose(ngmodel.prob('do', 'I', log=True), math.log(1/3))

    tuples = (
        ('I', 'am'),
        ('am', 'Sam'),
        ('I', 'am', 'Sam'),
        ('I', 'do', 'not', 'like', 'x')
    )

    for tup in tuples:
        args = (tup[1:], tup[0])
        assert ngmodel.prob(tup, log=False) == ngmodel.prob(*args, log=False)

    assert ngmodel.predict(('I', 'am'), return_prob=0) == 'Sam'
    assert ngmodel.predict(('I', 'am'), return_prob=1) == ('Sam', 0.5)
    assert math.isclose(ngmodel.predict(('I', 'am'), return_prob=2)[1], math.log(0.5))

    seed_tokens = (
        ('I', 'am'),
        ('I', 'do'),
        ('I', 'do', 'not'),
        ('I', 'do', 'x'),
        ('I', 'do', 'x', 'y'),
        None,
        (SPECIAL_TOKENS[SENT_START], ),
        SPECIAL_TOKENS[SENT_START],
        'I',
        'x'
    )
    for seed in seed_tokens:
        generated = list(ngmodel.generate_sequence(seed))
        assert len(generated) > 0
        assert all(isinstance(t, str) for t in generated)
        assert generated[-1] == SPECIAL_TOKENS[SENT_END]


@given(fit_corpus=st.booleans(),
       n=st.integers(1, 10),
       add_k_smoothing=st.floats(0.0, 2.0),
       keep_vocab=st.one_of(st.none(), st.integers(1, 100), st.floats(0.1, 1.0)),
       tokens_as_hashes=st.booleans())
def test_fit(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab, tokens_as_hashes):
    tokens_as_hashes = tokens_as_hashes and fit_corpus
    ng, full_vocab, ng_vocab = _fit_model(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab,
                                          tokens_as_hashes)

    if isinstance(keep_vocab, int):
        assert ng_vocab < full_vocab
    elif isinstance(keep_vocab, float) and keep_vocab < 1.0:
        assert ng_vocab <= full_vocab
    else:
        assert ng_vocab == full_vocab

    if tokens_as_hashes:
        assert all(isinstance(t, int) for t in ng_vocab)
    else:
        assert all(isinstance(t, str) for t in ng_vocab)


@given(fit_corpus=st.booleans(),
       n=st.integers(1, 5),
       add_k_smoothing=st.floats(0.0, 2.0),
       keep_vocab=st.one_of(st.none(), st.integers(1, 100), st.floats(0.1, 1.0)),
       tokens_as_hashes=st.booleans(),
       backoff=st.booleans(),
       return_prob=st.integers(0, 2))
def test_predict(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab, tokens_as_hashes, backoff,
                 return_prob):
    tokens_as_hashes = tokens_as_hashes and fit_corpus
    ng, full_vocab, ng_vocab = _fit_model(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab,
                                          tokens_as_hashes)
    given_args = _generate_random_given_args(full_vocab, n)

    for g in given_args:
        if g is not None and len(g) < n-1:
            with pytest.raises(ValueError):
                ng.predict(g, backoff=backoff, return_prob=return_prob)
        else:
            res = ng.predict(g, backoff=backoff, return_prob=return_prob)
            if return_prob > 0:
                assert isinstance(res, tuple)
                assert len(res) == 2
                pred, prob = res
                if return_prob == 1:
                    assert 0 < prob <= 1
                else:
                    assert prob <= 0

                if pred is None:
                    assert prob == (1.0 if return_prob == 1 else 0.0)
            else:
                pred = res

            if tokens_as_hashes:
                assert isinstance(pred, int) or pred is None
            else:
                assert isinstance(pred, str) or pred is None


@given(fit_corpus=st.booleans(),
       n=st.integers(1, 5),
       add_k_smoothing=st.floats(0.0, 2.0),
       keep_vocab=st.one_of(st.none(), st.integers(1, 100), st.floats(0.1, 1.0)),
       tokens_as_hashes=st.booleans(),
       backoff=st.booleans(),
       until_n=st.integers(0, 20),   # always generate at most 20 tokens, otherwise runs endlessly
       until_token=st.booleans())
def test_generate_sequence(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab, tokens_as_hashes,
                           backoff, until_n, until_token):
    tokens_as_hashes = tokens_as_hashes and fit_corpus

    if tokens_as_hashes:
        special_tokens = set(SPECIAL_TOKENS.keys())
    else:
        special_tokens = set(SPECIAL_TOKENS.values())

    ng, full_vocab, ng_vocab = _fit_model(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab,
                                          tokens_as_hashes)
    given_args = _generate_random_given_args(full_vocab, n)

    if until_token and ng_vocab:
        until_token = random.choice(list(ng_vocab))
    else:
        until_token = None

    for g in given_args:
        if until_n < 1:
            with pytest.raises(ValueError):
                list(ng.generate_sequence(g, backoff=backoff, until_n=until_n, until_token=until_token))
        elif g is not None and len(g) < n-1:
            with pytest.raises(ValueError):
                list(ng.generate_sequence(g, backoff=backoff, until_n=until_n, until_token=until_token))
        else:
            res = list(ng.generate_sequence(g, backoff=backoff, until_n=until_n, until_token=until_token))

            assert len(res) <= until_n
            if len(res) < until_n and backoff:
                assert until_token is not None
                assert res[-1] == until_token

            assert all(isinstance(t, int if tokens_as_hashes else str) for t in res)
            assert all(t in ng_vocab | special_tokens for t in res)


def _fit_model(textdata_en, corpus_en, fit_corpus, n, add_k_smoothing, keep_vocab, tokens_as_hashes):
    ng = NGramModel(n=n, add_k_smoothing=add_k_smoothing, keep_vocab=keep_vocab, tokens_as_hashes=tokens_as_hashes)

    if fit_corpus:
        data = corpus_en
        full_vocab = vocabulary(data, sort=False, tokens_as_hashes=tokens_as_hashes)
    else:
        data = [doc.split(' ') for doc in textdata_en.values()]
        full_vocab = {t for doc in data for t in doc}

    ret = ng.fit(data)
    assert isinstance(ret, NGramModel)

    if tokens_as_hashes:
        special_tokens_set = {SENT_START, SENT_END, OOV}
    else:
        special_tokens_set = {SPECIAL_TOKENS[SENT_START], SPECIAL_TOKENS[SENT_END], SPECIAL_TOKENS[OOV]}
    ng_vocab = {g[0] for g in ng.ngram_counts_.keys() if len(g) == 1} - special_tokens_set

    return ng, full_vocab, ng_vocab


def _generate_random_given_args(full_vocab, n):
    given_args = [None]
    full_vocab_list = list(full_vocab)
    for _ in range(100):
        given_args.append(random.sample(full_vocab_list, random.randint(1, n)))

    return given_args
