
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def combos(trec_runs, strategy="sum", output=sys.stdout, max_docs=1000):
    """
        strategy: "sum", "max", "min", "anz", "mnz", "med"
        max_docs: can be either a single integer or a dict{qid,value}
    """
    dfs = []
    for t in trec_runs:
        dfs.append(t.run_data)

    # Merge all runs
    """
    merged = reduce(lambda left,right: pd.merge(left, right, right_on=["query","docid"], left_on=["query","docid"], how="outer",
        suffixes=("","_")), dfs)
    merged = merged[["query", "docid", "score", "score_"]]
    """

    if len(dfs) < 2:
        return
    merged = pd.merge(dfs[0], dfs[1], right_on =["query", "docid"] , left_on=["query", "docid"] , how = "outer", suffixes=("", "_"))
    merged = merged[["query", "q0", "docid", "score", "score_"]]

    for d in dfs[2:]:
        merged = pd.merge(merged, d, right_on=["query","docid"], left_on=["query","docid"], how="outer", suffixes=("","_"))
        merged = merged[["query", "q0", "docid", "score", "score_"]]

    #merged["query"] = merged["query"].astype(str).apply(lambda x:x.strip())
    #return merged

    # merged.fillna(0.0, inplace=True) <- not filling nan's. Instead, I am using np.nan* functions
    # TODO: add option to normalize values
    # TODO: add option to act on the rank of documents instead of their scores

    if strategy == "sum":
        merge_func = np.nansum
    elif strategy == "max":
        merge_func = np.nanmax
    elif strategy == "min":
        merge_func = np.nanmin
    elif strategy == "anz":
        merge_func = np.nanmean
    elif strategy == "mnz":
        def mnz(values):
            n_valid_entries = np.sum(~np.isnan(values))
            return np.nansum(values) * n_valid_entries
        merge_func = mnz
    elif strategy == "med":
        merge_func = np.nanmedian
    else:
        print("Unknown strategy %s. Options are: 'sum', 'max', 'min', 'anz', 'mnz'" % (strategy))
        return None

    merged["ans"] = merged[["score", "score_"]].apply(merge_func, raw=True, axis=1)
    merged.sort_values(["query", "ans"], ascending=[True,False], inplace=True)

    for topic in merged['query'].unique():
        merged_topic = merged[merged['query'] == topic]
        if type(max_docs) == dict:
            maxd = max_docs[topic]
            for rank, entry in enumerate(merged_topic[["docid","ans"]].head(maxd).values, start=1):
                output.write("%s Q0 %s %d %f comb_%s\n" % (str(topic), entry[0], rank, entry[1], strategy))
        else:
            for rank, entry in enumerate(merged_topic[["docid","ans"]].head(max_docs).values, start=1):
                output.write("%s Q0 %s %d %f comb_%s\n" % (str(topic), entry[0], rank, entry[1], strategy))

    return merged

def vector_space_fusion(trec_runs, output=sys.stdout, max_docs=1000):

    dfs = []
    for t in trec_runs:
        dfs.append(t.run_data)

    # Merge all runs
    merged = reduce(lambda left,right: pd.merge(left, right, right_on=["query","docid"], left_on=["query","docid"], how="outer",
        suffixes=("","_")), dfs)
    merged = merged[["query", "docid", "score", "score_"]]
    merged.fillna(0.0, inplace=True)

    topics = trec_runs[0].topics()
    for topic in topics:

        mtopic = merged[merged["query"] == topic]
        nbrs = NearestNeighbors(n_neighbors=mtopic.shape[0], algorithm='ball_tree').fit(mtopic[["score","score_"]])

        pivot = mtopic.ix[mtopic["score"].idxmax()][["score", "score_"]]
        dists, order = nbrs.kneighbors(pivot.reshape(1, -1))

        docs = mtopic["docid"].values[order[0]]
        scores = 1.0/ (dists + 0.1)

        # Writes out information for this topic
        for rank, (d, s) in enumerate(zip(docs, scores[0])[:max_docs], start=1):
            output.write("%s Q0 %s %d %f vector_space_fusion\n" % (str(topic), d, rank, s))


def reciprocal_rank_fusion(trec_runs, k=60, max_docs=1000, output=sys.stdout):
    """
        Implements a reciprocal rank fusion as define in
        ``Reciprocal Rank fusion outperforms Condorcet and individual Rank Learning Methods`` by Cormack, Clarke and Buettcher.

        Parameters:
            k: term to avoid vanishing importance of lower-ranked documents. Default value is 60 (default value used in their paper).
            output: a file pointer to write the results. Sys.stdout is the default.
    """

    topics = trec_runs[0].topics()

    for topic in sorted(topics):
        doc_scores = {}
        for r in trec_runs:
            docs_for_run = r.get_top_documents(topic, n=1000)

            for pos, docid in enumerate(docs_for_run, start=1):
                doc_scores[docid] = doc_scores.get(docid, 0.0) + 1.0 / (k + pos)

        # Writes out information for this topic
        for rank, (docid, score) in enumerate(sorted(doc_scores.iteritems(), key=lambda x:(-x[1],x[0]))[:max_docs], start=1):
            output.write("%s Q0 %s %d %f reciprocal_rank_fusion_k=%d\n" % (str(topic), docid, rank, score, k))


def rank_biased_precision_fusion(trec_runs, p=0.80, max_docs=1000, output=sys.stdout):
    """
        Implements a rank biased precision (RBP) fusion

        Parameters:
            p: persistence parameter of RBP (default = 0.80)
            output: a file pointer to write the results. Sys.stdout is the default.
    """
    topics = trec_runs[0].topics()

    for topic in sorted(topics):
        doc_scores = {}
        for r in trec_runs:
            docs_for_run = r.get_top_documents(topic, n=1000)

            for pos, docid in enumerate(docs_for_run, start=1):
                doc_scores[docid] = doc_scores.get(docid, 0.0) + (1.0-p) * (p) ** (pos-1)

        # Writes out information for this topic
        for rank, (docid, score) in enumerate(sorted(doc_scores.iteritems(), key=lambda x:(-x[1],x[0]))[:max_docs], start=1):
            output.write("%s Q0 %s %d %f rank_biased_precision_fusion_p=%.3f\n" % (str(topic), docid, rank, score, p))


def borda_count(trec_runs):
    print "TODO: BordaCount (Aslam & Montague, 2001)"

def svp(trec_runs):
    print "TODO: (Gleich & Lim, 2011)"

def mpm(trec_runs):
    print "TODO: (Volkovs & Zemel, 2012) ---> probably it is not the case."

def plackeettluce(trec_runs):
    print "TODO: PlackettLuce (Guiver & Snelson, 2009)"






