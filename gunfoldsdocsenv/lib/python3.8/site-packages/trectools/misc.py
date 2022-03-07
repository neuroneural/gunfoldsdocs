# My library
from trectools import TrecPool, TrecRun

# Standard libraries
import string
import re

# TODO: use logging properly
import logging

# External libraries
import numpy as np
import pandas as pd
import scipy as sp

def remove_punctuation(text):
    t = re.sub('[' + re.escape(''.join(string.punctuation)) + ']', ' ', text)
    return re.sub(' +',' ',t)

def check_fleish_kappa(tuple_of_judgements):
    items = set()
    categories = set()
    n_ij = {}
    n = len(tuple_of_judgements)
    for judgement in tuple_of_judgements:
        for doc, rel in zip(range(judgement.shape[0]), judgement):
            items.add(doc)
            categories.add(rel)
            n_ij[(doc, rel)] = n_ij.get((doc, rel), 0) + 1
    N = len(items)
    p_j = {}
    for c in categories:
        p_j[c] = sum(n_ij.get((i,c), 0) for i in items) / (1.0*n*N)

    P_i = {}
    for i in items:
        P_i[i] = (sum(n_ij.get((i,c), 0)**2 for c in categories)-n) / (n*(n-1.0))

    P_bar = sum(P_i.itervalues()) / (1.0*N)
    P_e_bar = sum(p_j[c]**2 for c in categories)

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa

def unique_documents(list_of_runs, cutoff=10):
    # TODO: this should return a <RUN, [documents] >, in which for each RUN, we have a list
    # of documents that were uniquely provided by this RUN
    pass

def make_pool_from_files(filenames, strategy="topX", topX=10, rbp_strategy="sum", rbp_p=0.80, rrf_den=60):
    """
        Creates a pool object (TrecPool) from a list of filenames.
        ------
        strategy = (topX, rbp, rrf). Default: topX

        * TOP X options:
        topX = Integer Value. The number of documents per query to make the pool.

        * RBP options:
        topX = Integer Value. The number of documents per query to make the pool. Default 10.
        rbp_strategy = (max, sum). Only in case strategy=rbp. Default: "sum"
        rbp_p = A float value for RBP's p. Only in case strategy=rbp. Default: 0.80

        * RRF options:
        rrf_den = value for the Reciprocal Rank Fusion denominator. Default: 60
    """

    runs = []
    for fname in filenames:
        runs.append(TrecRun(fname))
    return make_pool(runs, strategy, topX=topX, rbp_p=rbp_p, rbp_strategy=rbp_strategy, rrf_den=rrf_den)


def make_pool(list_of_runs, strategy="topX", topX=10, rbp_strategy="sum", rbp_p=0.80, rrf_den=60):
    """
        Creates a pool object (TrecPool) from a list of runs.
        ------
        strategy = (topX, rbp). Default: topX
        topX = Integer Value. The number of documents per query to make the pool.
        rbp_strategy = (max, sum). Only in case strategy=rbp. Default: "sum"
        rbp_p = A float value for RBP's p. Only in case strategy=rbp. Default: 0.80
    """

    if strategy == "topX":
        return make_pool_topX(list_of_runs, cutoff=topX)
    elif strategy == "rbp":
        return make_pool_rbp(list_of_runs, topX=topX, p=rbp_p, strategy=rbp_strategy)
    elif strategy == "rrf":
        return make_pool_rrf(list_of_runs, topX=topX, rrf_den=rrf_den)

def make_pool_rrf(list_of_runs, topX=500, rrf_den=60):
    """
        topX = Number of documents per query. Default: 500.
        rrf_den = Value for the Reciprocal Rank Fusion denominator. Default is 60 as in the original paper:
        Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. G. V. Cormack. University of Waterloo. Waterloo, Ontario, Canada.
    """

    big_df = pd.DataFrame(columns=["query","docid","rbp_value"])

    for run in list_of_runs:
        df = run.run_data.copy()
        # NOTE: Everything is made based on the rank col. It HAS TO start by '1'
        df["rrf_value"] = 1.0 / (rrf_den + df["rank"])
        # Concatenate all dfs into a single big_df
        big_df = pd.concat((big_df,df[["query","docid","rrf_value"]]))

    # Default startegy is the sum.
    grouped_by_docid = big_df.groupby(["query","docid"])["rrf_value"].sum().reset_index()

    # Sort documents by rbp value inside each qid group
    grouped_by_docid.sort_values(by=["query","rrf_value"], ascending=[True,False], inplace=True)

    # Selects only the top X from each query
    result = grouped_by_docid.groupby("query").head(topX)

    # Transform pandas data into a dictionary
    pool = {}
    for row in result[["query", "docid"]].itertuples():
        q = int(row.query)
        if q not in pool:
            pool[q] = set([])
        pool[q].add(row.docid)

    return TrecPool(pool)


def make_pool_rbp(list_of_runs, topX = 100, p=0.80, strategy="sum"):
    """
        p = A float value for RBP's p. Default: 0.80
        Strategy = (max, sum). Default: "sum"
        topX = Number of documents per query to be used in the pool. Default: 100
    """

    big_df = pd.DataFrame(columns=["query","docid","rbp_value"])

    for run in list_of_runs:
        df = run.run_data.copy()
        # NOTE: Everything is made based on the rank col. It HAS TO start by '1'
        df["rbp_value"] = (1.0-p) * (p) ** (df["rank"]-1)
        # Concatenate all dfs into a single big_df
        big_df = pd.concat((big_df,df[["query","docid","rbp_value"]]))

    # Choose strategy for merging the different runs.
    if strategy == "sum":
        grouped_by_docid = big_df.groupby(["query","docid"])["rbp_value"].sum().reset_index()
    elif strategy == "max":
        grouped_by_docid = big_df.groupby(["query","docid"])["rbp_value"].max().reset_index()
    else:
        print("Strategy '%s' does not exist. Options are 'sum' and 'max'" % (strategy))

    # Sort documents by rbp value inside each qid group
    grouped_by_docid.sort_values(by=["query","rbp_value"], ascending=[True,False], inplace=True)

    # Selects only the top X from each query
    result = grouped_by_docid.groupby("query").head(topX)

    # Transform pandas data into a dictionary
    pool = {}
    for row in result[["query", "docid"]].itertuples():
        q = int(row.query)
        if q not in pool:
            pool[q] = set([])
        pool[q].add(row.docid)

    return TrecPool(pool)

def make_pool_topX(list_of_runs, cutoff=10):
    pool_documents = {}
    if len(list_of_runs) == 0:
        return TrecPool(pool_documents)

    topics_seen = set([])
    for run in list_of_runs:
        topics_seen = topics_seen.union(run.topics())
        for t in topics_seen:
            if t not in pool_documents.keys():
                pool_documents[t] = set([])
            pool_documents[t] = pool_documents[t].union(run.get_top_documents(t, n=cutoff))

    return TrecPool(pool_documents)

def sort_systems_by(list_trec_res, metric="map"):
    r = []
    for system in list_trec_res:
        # TODO: check for exceptions
        r.append((system.get_result(metric), system.get_runid()))

    # Sorting twice to have first the values sorted descending and then the run name sorted ascending
    return sorted(sorted(r,key=lambda x:x[1]), key=lambda x:x[0], reverse=True)


def get_correlation(sorted1, sorted2, correlation="kendall"):
    """
    Use sort_trec_res_by twice to obtain two <"value", "system"> list of results (sorted1 and sorted2)
    before using this method.
    Correlations implemented: kendalltau, pearson, spearman
    """
    def tau_ap(list1, list2):
        # List2 is the ground truth and list 1 is the list that we want to compare the tau_ap with.

        N = len(list1)
        # calculate C(i)
        c = [0] * N # C = [0,0,0,0,0,0,....]
        for i, element in enumerate(list1[1:]):
            # c[i] = number of items above rank i and correctly ranked w.r.t.. the item at rank i in list1
            #print "Checking element", element, " ranking", i + 1
            index_element_in_2 = list2.index(element)

            for other_element in list1[:i+1]:
                #print "Other element", other_element
                index_other_in_2 = list2.index(other_element)
                if index_element_in_2 > index_other_in_2 or other_element == element: # Check if it is correctly ranked
                    c[i] += 1
            #print "C[",i + 2,"]=", c[i]

        summation = 0
        for i in range(1,N):
            summation +=  (1. * c[i-1] / (i))
            #print c[i+1], (i)
        p = 1. / (N-1) * summation
        #print "P", p
        return (2 * p - 1., -1)

    if len(sorted1) != len(sorted2):
        print("ERROR: Arrays must have the same size. Given arrays have size (%d) and (%d)." % (len(sorted1), len(sorted2)))
        return np.nan

    # Transform a list of names into a list of integers
    s1 = zip(*sorted1)[1]
    s2 = zip(*sorted2)[1]
    m = dict(zip(s1, xrange(len(s2))))
    new_rank = []
    for s in s2:
        new_rank.append(m[s])

    if correlation  == "kendall" or correlation == "kendalltau":
        return sp.stats.kendalltau(xrange(len(s1)), new_rank)
    elif correlation  == "pearson" or correlation == "spearmanr":
        return sp.stats.pearsonr(xrange(len(s1)), new_rank)
    elif correlation  == "spearman" or correlation == "spearmanr":
        return sp.stats.spearmanr(xrange(len(s1)), new_rank)
    elif correlation  == "tauap" or correlation == "kendalltauap" or correlation == "tau_ap":
        return tau_ap(new_rank, range(len(s1)))
    else:
        print("Correlation %s is not implemented yet. Options are: kendall, pearson, spearman, tauap." % (correlation))
        return None


def confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

