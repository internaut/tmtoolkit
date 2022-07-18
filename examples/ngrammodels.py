from collections import Counter

from tmtoolkit.corpus import Corpus, doc_tokens
from tmtoolkit.tokenseq import token_ngrams
from tmtoolkit.utils import flatten_list


OOV = 0
SENT_START = 10
SENT_END = 11


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts_ = Counter()
        self.ngram_prob_ = {}  # np.array([], dtype=np.float_)

    def fit(self, corp):
        pad = self.n - 1
        unigram_sents = []
        for sents in doc_tokens(corp, tokens_as_hashes=True, sentences=True).values():
            new_sents = []
            for s in sents:
                if s:
                    new_sents.append([SENT_START] * pad + s + [SENT_END] * pad)
                else:
                    new_sents.append([])
            unigram_sents.extend(new_sents)

        self.ngram_counts_ = Counter()
        for i in range(1, self.n+1):
            ngrms_i = []
            for sent in unigram_sents:
                if i == 1:
                    ngrms_i.extend(map(lambda x: (x, ), sent))
                else:
                    ngrms_i.extend(token_ngrams(sent, n=i, join=False, ngram_container=tuple))
            self.ngram_counts_.update(ngrms_i)

        self.ngram_prob_ = {}
        n_unigrams = sum(c for ng, c in self.ngram_counts_.items() if len(ng) == 1)
        for ng, c in self.ngram_counts_.items():
            n = len(ng)
            tok, given = ng[(n - 1):], ng[:(n - 1)]
            assert len(tok) == 1

            if n == 1:
                p = c / n_unigrams
                lookup = tok[0]
            else:
                p = c / self.ngram_counts_[given]
                lookup = (tok[0], given)

            assert 0 < p <= 1
            self.ngram_prob_[lookup] = p


        # only works with bigrams so far:
        #self.ngram_prob = np.array([n/self.unigram_counts[ng[0]] for ng, n in self.ngram_counts.items()])
        #self.ngram_prob = {ng: n / self.unigram_counts[ng[0]] for ng, n in self.ngram_counts.items()}




# corp = Corpus.from_builtin_corpus('en-parlspeech-v2-sample-houseofcommons', sample=10, max_workers=1.0)
# corp

corp = Corpus({'d1': 'I am Sam', 'd2': 'Sam I am', 'd3': 'I do not like green eggs and ham'}, language='en')
corp

ngmodel = NGramModel(2)
ngmodel.fit(corp)


ngmodel.ngram_prob_[corp.nlp.vocab.strings['I'], (SENT_START, )]
ngmodel.ngram_prob_[SENT_END, (corp.nlp.vocab.strings['Sam'], )]
ngmodel.ngram_prob_[corp.nlp.vocab.strings['Sam'], (SENT_START, )]
ngmodel.ngram_prob_[corp.nlp.vocab.strings['Sam'], (corp.nlp.vocab.strings['am'], )]

#ng_docs = ngrams(corp, n, tokens_as_hashes=True, sentences=True, join=False)
#ng_docs = ngrams_augment_sents(ng_docs, n)

