# Standard libraries
# from subprocess import call # TODO: change os.system to subprocess
# TODO: use logging properly
import logging
import os

# External libraries
import pandas as pd

from scipy.stats import ttest_rel


'''
'''
class TrecRes:

    def __init__(self, filename=None):
        if filename:
            self.read_res(filename)
        else:
            self.filename = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.filename:
            return "Data from file %s" % (self.get_full_filename_path())
        else:
            return "Data file not set yet"

    def get_full_filename_path(self):
        return os.path.realpath(os.path.expanduser(self.filename))

    def get_filename(self):
        return os.path.basename(self.get_full_filename_path())

    def read_res(self, filename, result_header=["metric", "query", "value"], double_values=True):
        if len(result_header) != 3:
            print("ERROR: the header of your file should have size 3, but I just read %d colunms." % (len(result_header)))

        self.filename = filename
        self.data = pd.read_csv(filename, sep="\s+", names=result_header)
        #self.runid = self.data[self.data["metric"] == 'runid']["value"].get_values().at[-1] # TODO: replace by at or iat
        self.runid = "Anyone"

        if double_values:
            self.data = self.data[ self.data["metric"] != 'runid']
            self.data["value"] = self.data["value"].astype(float)

    def get_runid(self):
        return self.runid

    def compare_with(self, another_res, metric="P_10"):
        """
            Compare results with results of another run with a t-test.

            Returns the ttest_rel result
        """
        a = pd.Series(self.get_results_for_metric(metric))
        b = pd.Series(another_res.get_results_for_metric(metric))
        merged = pd.concat((a,b), axis=1)
        if merged.isnull().any().sum() > 0:
            merged = merged.dropna()
            print("The results do not share the same topics. Evaluating results on %d topics." % (merged.shape[0]))
        return ttest_rel(merged[0], merged[1])

    def get_result(self, metric="P_10", query="all"):
        # TODO: Use at[] and iat[] get_value -- http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.get_value.html
        if metric not in self.data["metric"].unique():
            print("Metric %s was not found" % (metric))
            return None
        v = self.data[(self.data["metric"] == metric) & (self.data["query"] == query)]["value"]
        if v.shape[0] == 0:
            print("Could not find any result using metric %s and query %s" % (metric, query))
            return None
        return v.values[0]

    def get_results_for_metric(self, metric="P_10"):
        '''
            Get the results in a map<query, value> for a giving metric.
        '''
        data_slice = self.data[self.data["metric"] == metric]

        # We have more than one query, we can ignore the "all" rows
        if data_slice.shape[0] > 1:
            data_slice = data_slice[data_slice["query"] != "all"]

        r = data_slice.to_dict(orient='list')
        return dict(zip(r["query"], r["value"]))

    def printresults(self, outputfilename, outputformat="csv", perquery=False):
        """
            outputformat options are 'trec' and 'csv'
        """
        if outputformat == 'csv':
            self.data.pivot("query", "metric", "value").to_csv(outputfilename)
        else:
            print("TODO: outputformat %s is not yet available" % (outputformat))


