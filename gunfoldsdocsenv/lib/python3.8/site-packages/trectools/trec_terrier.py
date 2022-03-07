# External libraries
import sarge
import os

from trectools import TrecRun

class TrecTerrier:

    def __init__(self, bin_path):
        self.bin_path = bin_path

    def run(self, index, topics, debug=True, model="PL2", ndocs=1000, result_dir=None, result_file="trec_terrier.run", terrierc=None, qexp=False, expTerms=5, expDocs=3, expModel="Bo1", showoutput=False):

        if result_dir is None:
            # Current dir is used if result_dir is not set
            result_dir = os.getcwd()

        cmd = "%s -r -Dtrec.topics=%s -Dtrec.model=%s -Dtrec.results=%s -Dtrec.results.file=%s" % (self.bin_path, topics, model,
                result_dir, result_file)

        cmd += " -Dmatching.retrieved_set_size=%d -Dtrec.output.format.length=%d " % (ndocs,ndocs)

        if terrierc is not None:
            cmd += " -c %d " % (terrierc)

        if qexp == True:
            cmd += " -q -Dexpansion.terms=%d -Dexpansion.documents=%d -Dtrec.qe.model=%s" % (expTerms, expDocs, expModel)

        if showoutput == False:
            cmd += (" > %s 2> %s" % (os.devnull, os.devnull))

        if debug:
            print("Running: %s " % (cmd))

        r = sarge.run(cmd).returncode

        if r == 0:
            return TrecRun(os.path.join(result_dir, result_file))
        else:
            print("ERROR with command %s" % (cmd))
            return None

#tt = TrecTerrier(bin_path="/data/palotti/terrier/terrier-4.0-trec-cds/bin/trec_terrier.sh")
#tr = tt.run(index="/data/palotti/terrier/terrier-4.0-trec-cds/var/index", topics="/data/palotti/trec_cds/metamap/default_summary.xml.gz", qexp=False)


