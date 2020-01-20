"""
(c) Ayan Sengupta - 2020
License: MIT License

This implementation of modified from sklearn's base estimator

Base classes for all estimators
"""

# Author: Ayan Sengupta

import warnings
from collections import defaultdict
import inspect
from sklearn.utils.validation import check_is_fitted, check_non_negative, check_random_state, check_array
import numpy as np

class BaseEstimator:
    """Base class for all estimators in JST
    
    Parameters
    ----------
    n_topic_components : int, optional (default=10)
        Number of topics.
    n_sentiment_components : int, optional (default=5)
        Number of sentiments.
    doc_topic_prior : float, optional (default=None)
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_topic_components`. Applicable for RJST, TSM and sLDA models.
    doc_sentiment_prior : float, optional (default=None)
        Prior of document sentiment distribution `theta`. If the value is None,
        defaults to `1 / n_sentiment_components`. Applicable for JST model.
    doc_topic_sentiment_prior : float, optional (default=None)
        Prior of document topic-sentiment distribution `pi`. If the value is None,
        defaults to `1 / n_sentiment_components`. Applicable for RJST, TSM and sLDA models.
    doc_sentiment_topic_prior : float, optional (default=None)
        Prior of document topic-sentiment distribution `pi`. If the value is None,
        defaults to `1 / n_topic_components`. Applicable for only JST model.
    topic_sentiment_word_prior : float, optional (default=None)
        Prior of topic-sentiment word distribution `beta`. If the value is None, defaults
        to `1 / (n_topic_components * n_sentiment_components)`.
    max_iter : integer, optional (default=10)
        The maximum number of iterations for Gibbs sampling.
    prior_update_step: integer, optional (default=5)
        How often to update priors using Minka's fixed point iteration
    evaluate_every : int, optional (default=0)
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.
    verbose : int, optional (default=0)
        Verbosity level.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    components_ : array, [n_features, n_topic_components, n_sentiment_components]
        topic-sentiment word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j, k]`` can be viewed as pseudocount that represents the
        number of times word `i` was assigned to topic `j` and sentiment `k`.
        It can also be viewed as distribution over the words for each topic-sentiment pair
        after normalization:
        ``model.components_ / model.components_.sum(axis=0)[np.newaxis,:,:]``.
    doc_topic_prior_ : float
        Prior of document topic distribution `theta`. If the value is None,
        it is `1 / n_topic_components`. Applicable for RJST, TSM and sLDA models.
    doc_sentiment_prior_ : float
        Prior of document sentiment distribution `theta`. If the value is None,
        it is `1 / n_sentiment_components`. Applicable for JST model.
    doc_topic_sentiment_prior_ : float
        Prior of document-topic-sentiment distribution `pi`. If the value is None,
        it is `1 / n_sentiment_components`. Applicable for RJST, TSM and sLDA models.
    doc_sentiment_topic_prior_ : float
        Prior of document-sentiment-topics distribution `pi`. If the value is None,
        it is `1 / n_topic_components`. Applicable for JST model.
    topic_sentiment_word_prior_ : float
        Prior of topic-sentiment-word distribution `beta`. If the value is None, it is
        `1 / (n_topic_components * n_sentiment_components)`.
    Examples
    --------
    >>> from jointtsmodels import RJST
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.datasets import fetch_20newsgroups
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
    >>> data = data[:1000]
    >>> vectorizer = CountVectorizer(max_df=0.7, min_df=10,
                                max_features=5000,
                                stop_words='english')
    >>> X = vectorizer.fit_transform(data)
    >>> vocabulary = vectorizer.get_feature_names()
    >>> inv_vocabulary = dict(zip(vocabulary,np.arange(len(vocabulary))))
    >>> lexicon_data = pd.read_excel('../lexicon/prior_sentiment.xlsx')
    >>> lexicon_data = lexicon_data.dropna()
    >>> lexicon_dict = dict(zip(lexicon_data['Word'],lexicon_data['Sentiment']))
    >>> model = RJST(n_topic_components=5,n_sentiment_components=5,
    ...     random_state=0)
    >>> model.fit(X.toarray(), lexicon_dict)
    RJST(...)
    >>> # get topics for some given samples:
    >>> model.transform()[:2]
    array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    >>> top_words = list(model.getTopKWords(vocabulary).values())
    >>> coherence_score_uci(X.toarray(),inv_vocabulary,top_words)
    1.107204574754555
           
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("JST estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn(FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __init__(self, n_topic_components=10, n_sentiment_components=5, doc_topic_prior=None, doc_sentiment_prior=None,
                 doc_topic_sentiment_prior=None, doc_sentiment_topic_prior=None,
                 topic_sentiment_word_prior=None, max_iter=10,
                 prior_update_step=5, evaluate_every=1, verbose=1, random_state=None):
        self.n_topic_components = n_topic_components
        self.n_sentiment_components = n_sentiment_components
        self.doc_topic_prior = doc_topic_prior
        self.doc_sentiment_prior = doc_sentiment_prior
        self.doc_topic_sentiment_prior = doc_topic_sentiment_prior
        self.doc_sentiment_topic_prior = doc_sentiment_topic_prior
        self.topic_sentiment_word_prior = topic_sentiment_word_prior
        self.max_iter = max_iter
        self.prior_update_step = prior_update_step
        self.evaluate_every = evaluate_every
        self.verbose = verbose
        self.random_state = random_state
        self.wordOccurenceMatrix = None
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
    def _check_params(self):
        """Check model parameters."""
        if self.n_topic_components <= 0:
            raise ValueError("Invalid 'n_topic_components' parameter: %r"
                             % self.n_topic_components)

        if self.n_sentiment_components <= 0:
            raise ValueError("Invalid 'n_sentiment_components' parameter: %r"
                             % self.n_sentiment_components)
                             
        if self.doc_topic_prior is not None and self.doc_topic_prior <= 0:
            raise ValueError("Invalid 'doc_topic_prior' parameter: %r"
                             % self.doc_topic_prior)
        
        if self.doc_sentiment_prior is not None and self.doc_sentiment_prior <= 0:
            raise ValueError("Invalid 'doc_sentiment_prior' parameter: %r"
                             % self.doc_sentiment_prior)
                             
        if self.doc_topic_sentiment_prior is not None and self.doc_topic_sentiment_prior <= 0:
            raise ValueError("Invalid 'doc_topic_sentiment_prior' parameter: %r"
                             % self.doc_topic_sentiment_prior)
                             
        if self.doc_sentiment_topic_prior is not None and self.doc_sentiment_topic_prior <= 0:
            raise ValueError("Invalid 'doc_sentiment_topic_prior' parameter: %r"
                             % self.doc_sentiment_topic_prior)
                             
        if self.topic_sentiment_word_prior is not None and self.topic_sentiment_word_prior <= 0:
            raise ValueError("Invalid 'topic_sentiment_word_prior' parameter: %r"
                             % self.topic_sentiment_word_prior)

    def _init_latent_vars(self):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = np.repeat(1. / self.n_topic_components, self.n_topic_components)
        else:
            self.doc_topic_prior_ = np.repeat(self.doc_topic_prior, self.n_topic_components)

        if self.doc_sentiment_prior is None:
            self.doc_sentiment_prior_ = np.repeat(1. / self.n_sentiment_components, self.n_sentiment_components)
        else:
            self.doc_sentiment_prior_ = np.repeat(self.doc_sentiment_prior, self.n_sentiment_components)
            
        if self.doc_topic_sentiment_prior is None:
            self.doc_topic_sentiment_prior_ = np.repeat(1. / self.n_sentiment_components, self.n_sentiment_components)
        else:
            self.doc_topic_sentiment_prior_ = np.repeat(self.doc_topic_sentiment_prior, self.n_sentiment_components)

        if self.doc_sentiment_topic_prior is None:
            self.doc_sentiment_topic_prior_ = np.repeat(1. / self.n_topic_components, self.n_topic_components)
        else:
            self.doc_sentiment_topic_prior_ = np.repeat(self.doc_sentiment_topic_prior, self.n_topic_components)
            
        if self.topic_sentiment_word_prior is None:
            self.topic_sentiment_word_prior_ = 1. / (self.n_topic_components * self.n_sentiment_components)
        else:
            self.topic_sentiment_word_prior_ = self.topic_sentiment_word_prior
            
    def _check_non_neg_array(self, X, whom):
        """check X format
        check X format and make sure no negative value in X.
        Parameters
        ----------
        X :  array-like
        """
        X = check_array(X)
        check_non_negative(X, whom)
        return X
        
    def fit(self, X, max_iter=None):
        """Learn model for the data X with Gibbs sampling.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Document word matrix.
        max_iter : int, optional (default=None)
        Returns
        -------
        self
        """
        raise NotImplementedError
    
    def conditionalDistribution(self, d, v):
        """
        Calculates the joint topic-sentiment probability for word v in document d
        Parameters
        -----------
        d: index
            Document index
        v: index
            Word index
        Returns
        ------------
        x: matrix
            Matrix (numTopics x numSentiments) of joint probabilities
        """
        raise NotImplementedError
        
    def _unnormalized_transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_topic_components)
            Document topic distribution for X. Applicable for RJST, TSM and sLDA.
        doc_sentiment_distr : shape=(n_samples, n_sentiment_components)
            Document sentiment distribution for X. Applicable for JST.
        """
        raise NotImplementedError
        
    def transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_topic_components)
            Document topic distribution for X. Applicable for RJST, TSM and sLDA.
        doc_sentiment_distr : shape=(n_samples, n_sentiment_components)
            Document sentiment distribution for X. Applicable for JST.
        """
        raise NotImplementedError
    
    def pi(self):
        """Document-topic-sentiment distribution according to fitted model.
        Returns
        -------
        doc_topic_sentiment_dstr : shape=(n_samples, n_topic_components, n_sentiment_components)
            Document-sentiment-topic distribution for X. Applicable for RJST, TSM and sLDA.
        doc_sentiment_topic_dstr : shape=(n_samples, n_sentiment_components, n_topic_components)
            Document-sentiment-topic distribution for X. Applicable for JST.
        """
        raise NotImplementedError
        
    def loglikelihood(self):
        """Calculate log-likelihood of generating the whole corpus
        Returns
        -----------
        Log-likelihood score: float
        """
        raise NotImplementedError
        
    def perplexity(self):
        """Calculate approximate perplexity for the whole corpus.
        Perplexity is defined as exp(-1. * log-likelihood per word)
        
        Returns
        ------------
        score : float
        """
        raise NotImplementedError
        
    def score(self):
        """Calculate log-likelihood of generating the whole corpus as score
        Returns
        -----------
        score: float
        """
        return self.loglikelihood()
        
    def getTopKWords(self, vocabulary, num_words=5):
        """
        Returns top num_words discriminative words for topic t and sentiment s based on topic_sentiment_word distribution
        Parameters
        ----------
        vocabulary : list
            list of vocabulary from vectorizer.
        num_words: int (default=5)
            number of words to be displayed for each topic-sentiment pair
        Returns
        -------
        worddict: dict
            Dictionary with (topic,sentiment) pair as key and list of top num_words words as value.
        """
        check_is_fitted(self)
        
        if len(vocabulary) != self.wordOccurenceMatrix.shape[1]:
            raise ValueError("Length of vocabulary does not match with document-word matrix fitted by model")
        
        pseudocounts = self.components_
        worddict = {}
        for t in range(self.n_topic_components):
            for s in range(self.n_sentiment_components):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(num_words + 1):-1]
                worddict[(t+1, s+1)] = [vocabulary[i] for i in topWordIndices]

        return worddict