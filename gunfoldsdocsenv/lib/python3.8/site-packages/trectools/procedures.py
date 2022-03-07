
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
from trectools import TrecRun
from trectools import misc
import os

def plot_system_rank(results, display_metric, outfile="plot.pdf"):
    """
        retults:
                the object outputed by get_results(...)
        display_metric:
                the human name of your metric. It will be written in the generated graph.
        outfile:
                a name for your plot
    """
    #rcParams.update({'figure.autolayout': True})

    # Transform data in a pandas df to process it easily
    df = pd.DataFrame(results, columns=["name","value","ci"])
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    # Get data
    values = df["value"]
    ci = df["ci"]
    teamnames = df["name"]
    X = df.index + 1

    # Prepares figure for plotting
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    # Small adjustments for plotting
    ax.set_ylabel(display_metric)
    ax.set_xlim(0.5,len(X)+0.5)

    plt.errorbar(X, values, fmt='o', yerr=ci)
    plt.xticks(X, teamnames, rotation='vertical')
    plt.savefig(outfile)
    return plt


def plot_distribuition():
    pass

def list_of_runs_from_path(path, suffix="*"):
    runs = []
    for r in glob(os.path.join(path, suffix)):
        tr = TrecRun(r)
        runs.append(tr)
    return runs

def evaluate_runs(trec_runs, trec_qrel, per_query):
    results = []
    for r in trec_runs:
        results.append(r.evaluate_run(trec_qrel, per_query))
    return results

def evaluate_runs_ubire(trec_runs, trec_qrel, trec_qread, extension):
    results = []
    for r in trec_runs:
        results.append(r.evaluate_ubire(trec_qrel, trec_qread, extension=extension))
    return results

def extract_metric_from_results(trec_ress, metric):
    results = []
    for res in trec_ress:
        rs = res.get_results_for_metric(metric).values()
        m = np.mean(rs)
        ci = misc.confidence_interval(rs, confidence=0.95)
        n = res.get_runid()

        results.append((n,m,ci))
    return results

def get_pool_coverage(trec_runs, trec_qrels, topX=10):
    results = []
    for r in trec_runs:
        #n = r.get_runid()
        n = r.run_data.get_value(0,"system") # TODO: Replace by at or iat
        covs = r.check_qrel_coverage(trec_qrels, topX=topX)

        m = np.mean(covs)
        ci = misc.confidence_interval(covs, confidence=0.95)

        results.append((n,m,ci))

    return results

