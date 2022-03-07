#!/usr/bin/env python
# encoding: utf-8

# Standard libraries
# from subprocess import call # TODO: change os.system to subprocess
# TODO: use logging properly
import logging
import os

# External libraries
import sarge
import pandas as pd
import numpy as np

from trectools import TrecRes

'''
'''
class TrecRun(object):
    def __init__(self, filename=None):
        if filename:
            self.read_run(filename)
        else:
            self.filename = None
            self.run_data = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.filename:
            return "Data from file %s" % (self.get_full_filename_path())
        else:
            return "Data file not set yet"

    def get_runid(self):
        return self.run_data["system"][0]

    def read_run(self, filename):
        self.run_data = pd.read_csv(filename, sep="\s+", names=["query", "q0", "docid", "rank", "score", "system"])
        # Make sure the values are correclty sorted by score
        self.run_data.sort_values(["query","score"], inplace=True, ascending=[True,False])
        self.filename = filename

    def get_full_filename_path(self):
        """
            Returns the full path of the run file.
        """
        return os.path.abspath(os.path.expanduser(self.filename))

    def get_filename(self):
        """
            Returns only the run file.
        """
        return os.path.basename(self.get_full_filename_path())

    def topics(self):
        """
            Returns a set with all topics.
        """
        return set(self.run_data["query"].unique())

    def topics_intersection_with(self, another_run):
        """
            Returns a set with topic from this run that are also in 'another_run'.
        """
        return self.topics().intersection(another_run.topics())

    def get_top_documents(self, topic, n=10):
        """
            Returns the top 'n' documents for a given 'topic'.
        """
        return list(self.run_data[self.run_data['query'] == topic]["docid"].head(n))

    def evaluate_run(self, trec_qrel_obj, per_query):
        from trectools import TrecEval
        evaluator = TrecEval(self, trec_qrel_obj)
        result = evaluator.evaluateAll(per_query)
        return result

    """
    def evaluate_external_script(self, cmd, debug=False):
        if debug:
            print("Running: %s " % (cmd))
        # TODO: if this command returns an error, I need to deal with it somehow
        sarge.run(cmd).returncode

    # def evaluate_my_trec_eval(self, q_trec_qrels):
    def evaluate_run(self, a_trec_qrel, outfile=None, printfile=True, debug=False):
        ""
            It is necessary to have trec_eval set on your PATH run this function.
        ""
        if printfile:
            if not outfile:
                outfile = self.get_full_filename_path() + ".res"
            cmd = "trec_eval -q %s %s > %s" % (a_trec_qrel.get_full_filename_path(), self.get_full_filename_path(), outfile)
            print("Running cmd: %s" % (cmd))
            self.evaluate_external_script(cmd, debug)
            return TrecRes(outfile)
        else:
            cmd = "trec_eval -q %s %s > .tmp_res" % (a_trec_qrel.get_full_filename_path(), self.get_full_filename_path())
            print("Running cmd: %s" % (cmd))
            self.evaluate_external_script(cmd, debug)
            res = TrecRes(".tmp_res")
            sarge.run("rm -f .tmp_res")
            return res

    def evaluate_ubire(self, a_trec_qrel, a_trec_other, p=0.8, stoprank=10, outfile=None, extension="ures",
                            printfile=True, debug=False):
        ""
            It is necessary to have ubire.jar set on your classpath to run this function.
        ""
        if not os.path.isfile(os.path.join(os.getcwd(), "ubire.jar")):
            print("File ubire.jar was not found in the current directory.")
            print("Please move it here (%s) and run this procedure again." % (os.getcwd()))
            return None

        if printfile:
            if not outfile:
                outfile = self.get_full_filename_path() + "." + extension

            cmd = "java -jar ubire.jar -q --qrels-file=%s --qread-file=%s --readability --rbp-p=%f --stoprank=%d --ranking-file=%s > %s" % (a_trec_qrel.get_full_filename_path(), a_trec_other.get_full_filename_path(), p, stoprank, self.get_full_filename_path(), outfile)
            self.evaluate_external_script(cmd, debug)
            return TrecRes(outfile)
        else:
            cmd = "java -jar ubire.jar -q --qrels-file=%s --qread-file=%s --readability --rbp-p=%f --stoprank=%d --ranking-file=%s > .tmp_ures" % (a_trec_qrel.get_full_filename_path(), a_trec_other.get_full_filename_path(), p, stoprank, self.get_full_filename_path())
            self.evaluate_external_script(cmd, debug)
            res = TrecRes(".tmp_ubire")
            sarge.run("rm -f .tmp_ubire")
            return res

    def evaluate_ndcg(self, a_trec_qrel, outfile=None, printfile=True, debug=False):
        ""
            It is necessary to have 'mygdeval.pl' set on your PATH run this function.
        ""
        if printfile:
            if not outfile:
                outfile = self.get_full_filename_path() + ".ndcg_res"
            cmd = "mygdeval.pl %s %s > %s" % (a_trec_qrel.get_full_filename_path(), self.get_full_filename_path(), outfile)
            self.evaluate_external_script(cmd, debug)
            return TrecRes(outfile)
        else:
            cmd = "mygdeval.pl %s %s > .tmp_ndcg_res" % (a_trec_qrel.get_full_filename_path(), self.get_full_filename_path())
            self.evaluate_external_script(cmd, debug)
            res = TrecRes(".tmp_ndcg_res")
            sarge.run("rm -f .tmp_ndcg_res")
            return res
    """
    def evaluate(self, metrics=["P@10", "P@100", "NDCG"]):
        pass

    def check_qrel_coverage(self, trecqrel, topX=10):
        """
            Check the average number of documents per topic that appears in
            the qrels among the topX documents of each topic.
        """
        covered = []
        for topic in sorted(self.topics()):
            cov = 0
            doc_list = self.get_top_documents(topic, topX)
            qrels_set = trecqrel.get_document_names_for_topic(topic)
            for d in doc_list:
                if d in qrels_set:
                    cov += 1
            covered.append(cov)
        return covered

    def get_mean_coverage(self, trecqrel, topX=10):
        """
            Check the average number of documents that appears in
            the qrels among the topX documents of each topic.
        """
        return np.mean(self.check_qrel_coverage(trecqrel, topX))

    def check_run_coverage(self, another_run, topX=10, debug=False):
        """
            Check the intersection of two runs for the topX documents.
        """
        runA = self.run_data[["query", "docid"]].groupby("query")[["query","docid"]].head(topX)
        runB = another_run.run_data[["query", "docid"]].groupby("query")[["query","docid"]].head(topX)

        common_topics = set(runA["query"].unique()).intersection(runB["query"].unique())

        covs = []
        for topic in common_topics:
            docsA = set(runA[runA["query"] == topic]["docid"].values)
            docsB = set(runB[runB["query"] == topic]["docid"].values)
            covs.append( len(docsA.intersection(docsB)) )

        if len(covs) == 0:
            print("ERROR: No topics in common.")
            return 0.0

        if debug:
            print("Evaluated coverage on %d topics: %.3f " % (len(common_topics), np.mean(covs)))
        return np.mean(covs)

    def print_subset(self, filename, topics):
        dslice = self.run_data[self.run_data["query"].apply(lambda x: x in set(topics))]
        dslice.sort_values(by=["query","score"], ascending=[True,False]).to_csv(filename, sep=" ", header=False, index=False)
        print("File %s writen." % (filename))

