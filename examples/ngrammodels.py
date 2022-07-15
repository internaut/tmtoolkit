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
        self.ngram_counts = Counter()
        self.ngram_prob = {}  # np.array([], dtype=np.float_)

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

        self.ngram_counts = Counter()
        for i in range(1, self.n+1):
            ngrms_i = []
            for sent in unigram_sents:
                if i == 1:
                    ngrms_i.extend(map(lambda x: (x, ), sent))
                else:
                    ngrms_i.extend(token_ngrams(sent, n=i, join=False, ngram_container=tuple))
            self.ngram_counts.update(ngrms_i)

        # only works with bigrams so far:
        #self.ngram_prob = np.array([n/self.unigram_counts[ng[0]] for ng, n in self.ngram_counts.items()])
        #self.ngram_prob = {ng: n / self.unigram_counts[ng[0]] for ng, n in self.ngram_counts.items()}




# corp = Corpus.from_builtin_corpus('en-parlspeech-v2-sample-houseofcommons', sample=10, max_workers=1.0)
# corp

corp = Corpus({'d1': 'I am Sam', 'd2': 'Sam I am', 'd3': 'I do not like green eggs and ham'}, language='en')
corp

ngmodel = NGramModel(2)
ngmodel.fit(corp)



ngmodel.ngram_counts
ngmodel.unigram_counts

#ng_docs = ngrams(corp, n, tokens_as_hashes=True, sentences=True, join=False)
#ng_docs = ngrams_augment_sents(ng_docs, n)

