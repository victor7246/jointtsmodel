"""
(c) Ayan Sengupta - 2020
License: MIT License

Utility functions
"""

import numpy as np
import scipy

def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    Parameter
    ----------
    alpha: array-like 
        Dirichlet distribution parameter
    Returns
    ----------
    x: array-like 
        sampled from dirichlet distribution
    """
    return np.random.dirichlet(alpha)


def sampleFromCategorical(theta):
    """
    Samples from a categorical/Multinomial distribution
    Parameter
    -----------
    theta: array-like
        Categorical/Multinomial sample
    Returns
    -----------
    x: int
        Sample from distribution
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    Parameter
    ----------
    alpha: array-like
        Categorical/Multinomial distribution
    K: int, optional
        Length of distribution vector
    Returns
    -----------
    x: int
        Log-Multinomial value of vector. Used in log-likelihood calculation.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(scipy.special.gammaln(alpha)) - scipy.special.gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * scipy.special.gammaln(alpha) - scipy.special.gammaln(K*alpha)

def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    Parameter
    ----------
    wordOccuranceVec: array-like
        Vectorized format of each document
    Returns
    ----------
    idx: index
        Index of each word in document
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx

def coherence_score_uci(X,inv_vocabulary,top_words):
    """
    Extrinsic UCI coherence measure
    Parameter
    ----------
    X : array-like, shape=(n_samples, n_features)
            Document word matrix.
    inv_vocabulary: dict
        Dictionary of index and vocabulary from vectorizer. 
    top_words: list
        List of top words for each topic-sentiment pair
    Returns
    -----------
    score: float
    """
    wordoccurances = (X > 0).astype(int)
    totalcnt = 0
    total = 0
    for allwords in top_words:
        for word1 in allwords:
            for word2 in allwords:
                if word1 != word2:
                    ind1 = inv_vocabulary[word1]
                    ind2 = inv_vocabulary[word2]
                    if ind1 > ind2:
                        total += np.log((wordoccurances.shape[0]*(np.matmul(wordoccurances[:,ind1],wordoccurances[:,ind2])+1))/(wordoccurances[:,ind1].sum()*wordoccurances[:,ind2].sum()))
                        totalcnt += 1
    return total/totalcnt
    
def coherence_score_umass(X,inv_vocabulary,top_words):
    """
    Extrinsic UMass coherence measure
    Parameter
    ----------
    X : array-like, shape=(n_samples, n_features)
            Document word matrix.
    inv_vocabulary: dict
        Dictionary of index and vocabulary from vectorizer. 
    top_words: list
        List of top words for each topic-sentiment pair
    Returns
    -----------
    score: float
    """
    wordoccurances = (X > 0).astype(int)
    totalcnt = 0
    total = 0
    for i in range(len(top_words)):
        allwords = topic_sentiment_df.top_words.iloc[i] #ast.literal_eval(topic_sentiment_df.top_words.iloc[i])
        for word1 in allwords:
            for word2 in allwords:
                if word1 != word2:
                    ind1 = inv_vocabulary[word1]
                    ind2 = inv_vocabulary[word2]
                    if ind1 > ind2:
                        total += np.log((np.matmul(wordoccurances[:,ind1],wordoccurances[:,ind2]) + 1)/np.sum(wordoccurances[:,ind1]))
                        totalcnt += 1
    return total/totalcnt
    
def symmetric_kl_score(pk,qk):
    """
    Symmetric KL divergence score between two probability distributions
    Parameter
    ----------
    pk: array-like
        Probability distribution
    qk: array-like
        Probability distribution
    Returns
    -----------
    score: float
        Symmetric KL divergence score between pk and qk
    """
    score = scipy.stats.entropy(pk,qk)*.5 + scipy.stats.entropy(qk,pk)*.5
    return score

def Hscore(transformedX, subsample=1.0):
    """
    H score on transformed matrix. H score is calculated using inter cluster KL divergence score and intra cluster KL divergence score.
    Reference
    ----------
        [1] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        
    Parameter
    ----------
    transformedX: array-like
        Probability distribution
    subsample: float, optional (default=1)
        subsample size for H score calculation
    Returns
    -----------
    score: float
    """
    subsample_size = int(transformedX.shape[0]*subsample)
    transformedX_ = transformedX[:subsample_size,:]
    
    n_components = transformedX.shape[1]
    
    all_kl_scores = scipy.spatial.distance.cdist(transformedX_, transformedX_, symmetric_kl_score)
    dt = (transformedX_ == transformedX_.max(axis=1, keepdims=True)).astype(int)

    intradist = 0
    for i in range(n_components):
        cnt = dt[:,i].sum()
        tmp = np.outer(dt[:,i],dt[:,i])
        tmp = tmp * all_kl_scores
        intradist += tmp.sum()*1.0/(cnt*(cnt-1))
    intradist = intradist/n_components
    
    interdist = 0
    for i in range(n_components):
       for j in range(n_components):
           if i != j:
             cnt_i = dt[:,i].sum()
             cnt_j = dt[:,j].sum()
             tmp = np.outer(dt[:,i], dt[:,j])
             tmp = tmp * all_kl_scores
             interdist += tmp.sum()*1.0/(cnt_i*cnt_j)
    interdist = interdist/(n_components*(n_components-1))

    return intradist/interdist