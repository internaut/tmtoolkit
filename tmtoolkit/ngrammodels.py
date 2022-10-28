"""
N-gram models as in [JurafskyMartin2021]_. Mainly provides the :class:`NGramModel` class.

.. [JurafskyMartin2021] Jurafsky, D. and Martin, J.H., 2021. Speech and Language Processing (3rd ed. draft). Online:
                        https://web.stanford.edu/~jurafsky/slp3/

"""

from __future__ import annotations
import math
import random
from collections import Counter
from typing import Optional, Union, List, Tuple, Generator, Dict, Iterable

from bidict import bidict

from tmtoolkit.corpus import doc_tokens, Corpus
from tmtoolkit.tokenseq import token_ngrams, token_hash_convert
from tmtoolkit.types import StrOrInt
from tmtoolkit.utils import flatten_list

OOV = 0
SENT_START = 10
SENT_END = 11


SPECIAL_TOKENS = bidict({
    SENT_START: '<s>',
    SENT_END: '</s>',
    OOV: '<oov>'
})


class NGramModel:
    def __init__(self, n: int, add_k_smoothing: float = 1.0, keep_vocab: Optional[Union[int, float]] = None,
                 tokens_as_hashes: bool = True):
        """
        Initialize an n-gram model with gram size `n`.

        :param n: strictly positive integer for the gram size
        :param add_k_smoothing: smoothing constant added to each count; must be positive
        :param keep_vocab: optional; specifies the maximum vocabulary size, i.e. keep only the most frequent tokens;
                           either int or float; if float, the number implicates the proportion of most frequent tokens
                           to keep
        :param tokens_as_hashes: if True, use token type hashes (integers) instead of textual representations (strings)
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError('`n` must be a strictly positive integer')

        if add_k_smoothing < 0:
            raise ValueError('`add_k_smoothing` must be positive')

        if keep_vocab is not None:
            if not isinstance(keep_vocab, (float, int)):
                raise ValueError('`keep_vocab` must be either a float or an int')

            if keep_vocab <= 0:
                raise ValueError('if `keep_vocab` is given, it must be strictly positive')

            if isinstance(keep_vocab, float) and keep_vocab > 1.0:
                raise ValueError('if `keep_vocab` is given as float, it must be in range (0, 1]')

        self.n: int = n
        self.k: float = add_k_smoothing
        self.keep_vocab: Optional[Union[int, float]] = keep_vocab
        self.tokens_as_hashes: bool = tokens_as_hashes

        self.vocab_size_: int = 0
        self.n_unigrams_: int = 0
        self.ngram_counts_: Counter = Counter()  # maps tuples of 1-grams, 2-grams, ..., n-grams to their count
        self.stringstore = None  # used for string <-> hash conversion when Corpus obj. was passed in `fit()`

    def __str__(self) -> str:
        """String representation of this NGramModel object."""
        return self.__repr__()

    def __repr__(self):
        """String representation of this NGramModel object."""
        return f'<NGramModel [n={self.n}, k={self.k}]>'

    def fit(self, corp: Union[Corpus, List[List[StrOrInt]]]) -> NGramModel:
        """
        Fit this n-gram model using a :class:`~tmtoolkit.corpus.Corpus` object or a list of token sequences.

        :param corp: a :class:`~tmtoolkit.corpus.Corpus` object or a list of token sequences used as training data
        :return: this instance
        """

        # check input
        self.stringstore = None  # reset

        if isinstance(corp, Corpus):
            self.stringstore = corp.nlp.vocab.strings
            corp = flatten_list(doc_tokens(corp, tokens_as_hashes=self.tokens_as_hashes, sentences=True).values())
        elif not isinstance(corp, list):
            raise ValueError('`corp` must be either a Corpus object or a list of sentences as token sequences')

        # apply padding, i.e. adding special sentence start and end tokens
        unigram_sents = list(map(self.pad_sequence, corp))

        # count the tokens (unigrams)
        unigram_counts = Counter(t for s in unigram_sents for t in s)

        # this is done before possible filtering when keep_vocab is set
        self.n_unigrams_ = sum(unigram_counts.values())

        if self.keep_vocab is not None:
            # keep only the most frequent tokens in the vocabulary
            if isinstance(self.keep_vocab, float):
                keep_n = round(len(unigram_counts) * self.keep_vocab)
            else:
                keep_n = self.keep_vocab

            # get the most freq. tokens
            keep_tok = set(list(zip(*unigram_counts.most_common(keep_n)))[0])

            # filter
            unigram_counts = {k: v for k, v in unigram_counts.items() if k in keep_tok}
        else:
            # keep the full vocabulary
            keep_tok = None

        # set vocab size
        self.vocab_size_ = len(unigram_counts)

        # count n-grams
        self.ngram_counts_ = Counter()
        oov_tok = OOV if self.tokens_as_hashes else SPECIAL_TOKENS[OOV]
        for i in range(1, self.n+1):   # get counts from unigram to n-gram; TODO: don't count unigrams again
            ngrms_i = []
            for sent in unigram_sents:
                if keep_tok:  # filter tokens
                    sent = [t if t in keep_tok else oov_tok for t in sent]
                # generate n-grams with n=i
                ngrms_i.extend(token_ngrams(sent, n=i, join=False, ngram_container=tuple))
            self.ngram_counts_.update(ngrms_i)

        return self

    def predict(self, given: Optional[Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]]] = None,
                backoff: bool = False, return_prob: int = 0) \
            -> Union[Optional[StrOrInt], Tuple[Optional[StrOrInt], float]]:
        """
        Predict the most likely continuation candidate (i.e. next token) given a sequence of tokens `given`. If
        `given` is None, assume a sentence start.

        :param given: optional given single token or given sequence of tokens; if None, assume a sentence start
        :param backoff: if True, then if no continuation candidates for the given sequence can be found, iteratively
                        back off to a smaller given sequence (eventually up until no given sequence) until continuation
                        candidates are found
        :param return_prob: 0 - don't return prob., 1 – return prob., 2 – return log prob.
        :return: if `return_prob` is 0, return the most likely next token; if `return_prob` is not zero, return a
                 2-tuple with ``(must likely token, prediction probability)``; if `backoff` is False and no
                 continuation candidates are found, return None or ``(None, 1.0)``
        """
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        given = self._prepare_given_param(given)  # turn `given` into a tuple

        # find the continuation candidates and their probabilities
        probs = self._probs_for_given(given, log=return_prob == 2, backoff=backoff)

        if probs:
            # sort by prob.
            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if return_prob != 0:
                return probs[0]     # return most likely next token and prob.
            else:
                return probs[0][0]  # return most likely next token
        else:  # no continuation candidates found
            if return_prob == 0:
                return None
            else:
                return None, 1.0 if return_prob == 1 else 0.0

    def generate_sequence(self, given: Optional[Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]]] = None,
                          backoff: bool = True, until_n: Optional[int] = None,
                          until_token: Optional[StrOrInt] = SENT_END) -> Generator[StrOrInt]:
        """
        Generate a random sequence of tokens given an optional "seed" sequence as `given`. If `given` is None,
        assume a sentence start. Generate the sequence until either a number of tokens `until_n` is reached or a
        certain token `until_token` was generated.

        The random sequence is generated by sampling from a probability distribution that is conditional on the previous
        ``n-1`` token(s) in the given n-gram model.

        This method is a generator that yields one token at a time.

        :param given: optional given single token or given sequence of tokens; if None, assume a sentence start
        :param backoff: if True, then if no continuation candidates for the given sequence can be found, iteratively
                        back off to a smaller given sequence (eventually up until no given sequence) until continuation
                        candidates are found
        :param until_n: if given, sample until at maximum this number of tokens is generated
        :param until_token: if given, sample until at maximum this certain token is generated
        :return: yields one token at a time
        """

        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        if not self.tokens_as_hashes and isinstance(until_token, int):
            until_token = SPECIAL_TOKENS[until_token]

        given = self._prepare_given_param(given)  # turn `given` into a tuple

        i = 0
        while True:  # generate the next token
            # get prob. for continuation candidates; can't use log prob. here since `random.choices` can't handle
            # log prob.
            probs = self._probs_for_given(given, log=False, backoff=backoff)

            if not probs:
                break

            # sample the next token
            x = random.choices(list(probs.keys()), list(probs.values()))[0]

            # move the `given` sequence this one token forward
            given = (given + (x, ))[1:]
            i += 1

            yield x

            # check stopping conditions
            if until_n is not None and i >= until_n:
                break

            if until_token is not None and x == until_token:
                break

    def prob(self, x: Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]],
             given: Optional[Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]]] = None, log: bool = True,
             pad_input: bool = False) -> float:
        """
        Return the probability of token or token sequence `x` optionally given a sequence `given`. If `given` is not
        None, it is simply prepended to `x`.

        For each token ``t_i`` in the concatenated sequence ``S`` of `given` and `x`, calculate the overall prob. of the
        sequence ``S`` in the n-gram model as ``prod(`P(t_i|t_{i-n+1}, ..., t_{i-1})`)`` (or the sum of the respective
        log. prob. if `log` is True).

        :param x:  single token or sequence of tokens
        :param given: optional given single token or given sequence of tokens
        :param log: if True, return log prob.
        :param pad_input: if True, pad `x` with sentence start token(s)
        :return: (log) probability
        """
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        if isinstance(x, list):
            x = tuple(x)

        if not isinstance(x, tuple):
            x = (x,)

        if not x:
            raise ValueError('at least one token must be given in `x`')

        if isinstance(given, list):
            given = tuple(given)

        if given is not None:
            # prepend `given` to input `x`
            if not isinstance(given, tuple):
                given = (given,)
            x = given + x

        if pad_input:
            x = self.pad_sequence(x, sides='left')

        # turn the input x into n-grams
        if len(x) > self.n:
            x = token_ngrams(x, self.n, join=False, ngram_container=tuple)
        else:
            x = [x]

        # sum up / multiply up the (log) prob. by using each n-gram `ng`
        p = 0 if log else 1
        for ng in x:
            # calculate `P(t_i|t_{i-n+1}, ..., t_{i-1})` where `t_i` is the last token in `ng` and the given part
            # `t_{i-n+1}, ..., t_{i-1}` are all the tokens before the last token
            p_ng = self._prob_smooth(ng, log=log)

            if log:
                p += p_ng
            else:
                p *= p_ng

        # check prob.
        if log:
            assert 0 <= math.exp(p) <= 1, 'smoothed prob. must be in [0, 1] interval'
        else:
            assert 0 <= p <= 1, 'smoothed prob. must be in [0, 1] interval'

        return p

    def perplexity(self, x: Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]], pad_input: bool = False) -> float:
        """
        Calculate the perplexity for a given single token or sequence of tokens `x`. The perplexity is defined as
        ``perplexity(x) = p(x)^-1/N``, where ``p(x)`` is the prob. of the sequence `x` as calc. by
        :meth:`~NGramModel.prob` and ``N`` is the vocabulary size.

        :param x:  single token or sequence of tokens
        :param pad_input: if True, pad `x` with sentence start token(s)
        :return: perplexity
        """

        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        if self.vocab_size_ <= 0:
            raise ValueError('vocabulary must be non-empty')

        log_p = self.prob(x, pad_input=pad_input)
        return math.pow(math.exp(log_p), -1.0/self.vocab_size_)

    def pad_sequence(self, s: Union[Tuple[StrOrInt, ...], List[StrOrInt]], sides: str = 'both') \
            -> Union[Tuple[StrOrInt, ...], List[StrOrInt]]:
        """
        Prepend start sentence token(s) and/or append end sentence token(s) to a sequence of tokens `s`. If `s` is an
        empty sequence don't apply padding.

        :param s: sequence of tokens
        :param sides: either 'left', 'right' or 'both'
        :return: padded sequence
        """
        if not isinstance(s, (tuple, list)):
            raise ValueError('`s` must be tuple or list')

        if sides not in {'left', 'right', 'both'}:
            raise ValueError("`sides` must be either 'left', 'right' or 'both'")

        pad = max(self.n - 1, 1)

        if sides in {'left', 'both'}:
            pad_l = pad
        else:
            pad_l = 0

        if sides in {'right', 'both'}:
            pad_r = pad
        else:
            pad_r = 0

        start_symbol = SENT_START if self.tokens_as_hashes else SPECIAL_TOKENS[SENT_START]
        end_symbol = SENT_END if self.tokens_as_hashes else SPECIAL_TOKENS[SENT_END]

        if s:
            if isinstance(s, tuple):
                s = list(s)

            # apply padding
            s_ = [start_symbol] * pad_l + s + [end_symbol] * pad_r

            if isinstance(s, tuple):
                return tuple(s_)
            else:
                return s_
        else:
            if isinstance(s, tuple):
                return tuple()
            else:
                return []

    def convert_token_sequence(self, tok: Iterable[StrOrInt], collapse: Optional[str] = ' ') \
            -> Union[str, Tuple[StrOrInt, ...], List[StrOrInt]]:
        """
        Convert a sequence of tokens `tok` to

        :param tok:
        :param collapse:
        :return:
        """
        if self.stringstore is None:
            raise ValueError('this method can only be used when this model was fit on a `Corpus` object or '
                             '`self.stringstore` was set manually')

        if not self.tokens_as_hashes and isinstance(collapse, str):
            collapse = None

        return token_hash_convert(tok, stringstore=self.stringstore, special_tokens=SPECIAL_TOKENS, collapse=collapse)

    def _prepare_given_param(self, given: Optional[Union[StrOrInt, Tuple[StrOrInt, ...], List[StrOrInt]]]) \
            -> Tuple[StrOrInt, ...]:
        """
        Helper function to "standardize" the `given` parameter by turning it into a sequence of one or more token(s).
        """
        if self.n == 1:  # for a unigram model, the given parameter is never used -> return empty tuple
            return tuple()

        if given is None:  # if given is not set, assume a sentence start
            given = (SENT_START, ) * (self.n - 1)
        else:
            if isinstance(given, list):  # turn a token list into a token tuple
                given = tuple(given)
            elif not isinstance(given, tuple):  # if a single token is given, wrap it with a tuple
                given = (given,)

            if len(given) > self.n - 1:
                given = given[-(self.n - 1):]
            elif len(given) < self.n - 1:
                raise ValueError(f'for a {self.n}-gram model you must provide `given` with at least {self.n-1} tokens')

        assert len(given) == self.n - 1

        return given

    def _prob_smooth(self, x: Tuple[StrOrInt, ...], log: bool) -> float:
        """
        Helper function to calculate the (optionally smoothed) probability for a token sequence `x`.

        Calculates `P(t_i|t_{i-n+1}, ..., t_{i-1})` where `t_i` is the last token in `x` and the given part
        `t_{i-n+1}, ..., t_{i-1}` are all the tokens before the last token.
        """

        n = len(x)
        assert isinstance(x, tuple), '`x` must be a tuple'
        assert 1 <= n <= self.n, f'`x` must be a tuple of length 1 to {self.n} in a {self.n}-gram model'

        # numerator: P(t_1, ..., t_n) for an n-gram token t_1 to t_n
        c = self.ngram_counts_.get(x, 0)

        # denominator: P(t_1, ..., t_{n-1})
        if n == 1:  # single token
            d = self.n_unigrams_
        else:       # x[:(self.n-1)] is the "given" sequence, i.e. the sequence before x[-1]
            d = self.ngram_counts_.get(x[:(self.n-1)], 0)

        if log:
            p = math.log(c + self.k) - math.log(d + self.k * self.vocab_size_)
            assert 0 <= math.exp(p) <= 1, 'smoothed prob. must be in [0, 1] interval'
        else:
            p = (c + self.k) / (d + self.k * self.vocab_size_)
            assert 0 <= p <= 1, 'smoothed prob. must be in [0, 1] interval'

        return p

    def _probs_for_given(self, given: Tuple[StrOrInt, ...], log: bool, backoff: bool = False) -> Dict[StrOrInt, float]:
        """
        Helper function to get continuation candidates for a given sequence of tokens `given`.
        """
        probs = {}
        len_g = len(given)

        while len_g >= 0:  # backoff loop
            for ng in self.ngram_counts_.keys():  # iterate through all n-grams in the training data
                if len(ng) == len_g + 1 and ng[:len_g] == given:  # this is an n-gram that starts with `given`
                    candidate = ng[-1]  # the candidate is the last part
                    assert candidate not in probs
                    # get the prob. of the candidate
                    probs[candidate] = self._prob_smooth(ng, log=log)

            if probs or not backoff:   # found candidates or not backing off
                break

            # backing off: try to find candidates using a smaller given sequence; cut of the first token
            # it's fine down to an empty given token – in that case we sample from all unigrams
            given = given[1:]
            len_g = len(given)

        return probs

