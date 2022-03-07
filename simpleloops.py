from gunfolds.tools.conversions import g2num, num2CG, graph2nx
from gunfolds.tools.bfutils import undersample
import networkx
from numpy import argsort



def simple_loops(g, u):
    """
    iterator over the list of simple loops of graph g at the undersample rate u
    """
    gx = graph2nx(num2CG(g2num(undersample(g, u)), len(g)))
    for l in networkx.simple_cycles(gx):
        yield l


def print_loops(g, u):
    l = [x for x in simple_loops(g, u)]
    lens = map(len, l)
    for i in argsort(lens):
        print l[i]
