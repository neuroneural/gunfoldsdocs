# External libraries
import sarge
import os

from trectools import TrecRun

#TODO: sourceforce was offline when I wrote this code. I need to get to the docs to check regarinding the baselines and how to specify them.
class TrecIndri:

    def __init__(self, bin_path):
        self.bin_path = bin_path

    def queryclarity(self, query, index):
        cmd = "%s/clarity -query='%s' -index=%s 2> %s | head -n1 | cut -f2 -d'=' | cut -d' ' -f3 " % (self.bin_path,query,index,os.devnull)
        p = sarge.run(cmd, stdout=sarge.Capture())
        try:
            f = float(p.stdout.text)
            return f
        except Exception as e:
            print('Query Clarity exception: %s' % (e))
            return 0.0

    def queryclarity_topics(self, trec_topics, index):
        results = {}
        for topid, top in trec_topics.topics.iteritems():
            r = self.queryclarity(top, index)
            results[topid] = r
        return results

    def run(self, index, topics, model="dirichlet", parameters={"mu":2500}, server=None, stopper=None, result_dir=None, result_file="trec_indri.run", ndocs=1000, qexp=False, expTerms=5, expDocs=3, showerrors=True, debug=True, queryOffset=1):

        if result_dir is None:
            # Current dir is used if result_dir is not set
            result_dir = os.getcwd()

        outpath = ""
        if result_dir is not None and result_file is not None:
            outpath = os.path.join(result_dir, result_file)
        elif result_file is not None:
            outpath = result_file

        cmd = "%s/IndriRunQuery %s -index=%s -trecFormat=true -queryOffset=%d " % (self.bin_path, topics, index, queryOffset)

        # Specify number of documents to retrieve
        cmd += " -count=%d " % (ndocs)


        if model == "dirichlet":
            if "mu" not in parameters:
                print("WARNING: no value was set to the parameter 'mu'. Using default value mu = 2500.")
                parameters["mu"] = 2500
            cmd += " -rule=method:dirichlet,mu:%d" % (parameters["mu"])

        elif model == "tfidf":
            if "k1" not in parameters:
                print("WARNING: no values were set to the parameter 'k1'. Using default value k1 = 1.2.")
                parameters["k1"] = 1.2
            if "b" not in parameters:
                print("WARNING: no values were set to the parameter 'b'. Using default value b = 0.75.")
                parameters["b"] = 0.75

            cmd += " -baseline=tfidf,k1:%f,b:%f" % (parameters["k1"], parameters["b"])

        elif model == "okapi":
            if "k1" not in parameters:
                print("WARNING: no values were set to the parameter 'k1'. Using default value k1 = 1.2.")
                parameters["k1"] = 1.2
            if "k3" not in parameters:
                print("WARNING: no values were set to the parameter 'k3'. Using default value k3 = 7.")
                parameters["k3"] = 7
            if "b" not in parameters:
                print("WARNING: no values were set to the parameter 'b'. Using default value b = 0.75.")
                parameters["b"] = 0.75

            cmd += " -baseline=okapi,k1:%f,b:%f,k3:%f" % (parameters["k1"], parameters["b"], parameters["k3"])
        else:
            print("ERROR: model %s is not yet implemented. Using default model")

        if "field" in parameters:
            cmd += "field:%s " % (parameters["field"])

        if server is not None:
            cmd += " -server=%s " % (server)

        if stopper is not None:
            cmd += " -stopper.word=%s " % (stopper)

        if qexp == True:
            cmd += " -fbDocs=%d -fbTerms=%d " % (expTerms, expDocs)

        if showerrors == True:
            cmd += (" > %s " % (outpath))
        else:
            cmd += (" 2> %s > %s "  % (os.devnull, outpath))

        if debug:
            print("Running: %s " % (cmd))

        r = sarge.run(cmd).returncode

        if r == 0:
            return TrecRun(os.path.join(result_dir, result_file))
        else:
            print("ERROR with command %s" % (cmd))
            return None

#tt = TrecIndri(bin_path="/data/palotti/terrier/terrier-4.0-trec-cds/bin/trec_terrier.sh")
#tr = tt.run(index="/data/palotti/terrier/terrier-4.0-trec-cds/var/index", topics="/data/palotti/trec_cds/metamap/default_summary.xml.gz", qexp=False)


