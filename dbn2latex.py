import sys

#from gunfolds.data.testgraphs import *
from bfutils import undersample
import ecj
#import load_data
import zickle
import igraph
import math
import numpy as np
import os
import scipy
from scipy import array, cos, sin, deg2rad, rad2deg
import io as StringIO

def graph2dict(g):
    D = {}
    for v in range(0, len(g.vs)):
        D[g.vs[v]["label"]] = {}
        for u in g.neighbors(v, mode="OUT"):
            D[g.vs[v]["label"]][g.vs[u]["label"]] = 1
    return D


def paintSCC(g, cm):
    nameidx = {}
    D = graph2dict(g)
    scc = ecj.scc(D)
    for i in range(0, len(scc)):
        for v in scc[i]:
            g.vs[g.vs['label'].index(v)]["color"] = cm[i]
fstr = "%.5f"  # precision to use


def dict2graph(D):
    A = scipy.zeros([len(D), len(D)])
    nodes = []
    indices = {}
    c = 0
    for v in D:
        nodes.append(v)
    idx = np.argsort([int(v) for v in nodes])
    nodes = [nodes[i] for i in idx]
    for v in nodes:
        indices[v] = c
        c += 1
    for v in nodes:
        for u in D[v]:
            if D[v][u] not in (2,3):
                A[indices[v], indices[u]] = 1
    g = igraph.Graph.Adjacency(A.tolist())
    g.vs["label"] = nodes
    return g


def matrix_start(mname='generic', w_gap=0.45, h_gap=0.5, stl=''):
    print("\matrix ("\
          + mname\
          + ") [matrix of nodes, row sep="\
          + str(h_gap)\
          + "cm,column sep="\
          + str(w_gap)\
          + "cm"\
          + stl\
          + "]")
    print("{")


def matrix_end():
    print("};")


def matrix_grid(G, s, mname='generic', w_gap=0.5, h_gap=0.5, type="obs", stl=''):
    matrix_start(mname=mname, w_gap=w_gap, h_gap=h_gap, stl=stl)
    keylist = G.keys()
    idx = np.argsort([int(v) for v in keylist])
    keylist = [keylist[i] for i in idx]
    for v in keylist:
        print(" ", "& ".join(["\\node[" + type + "]{" + v + "};" for i in range(1, s)]) + "\\\\")
    matrix_end()


def matrix_edges(G, s, mname='generic'):
    nodes = list(G.keys())
    idx = np.argsort([int(v) for v in nodes])
    nodes = [nodes[i] for i in idx]
    for v in nodes:
        idx1 = nodes.index(v) + 1
        for u in G[v]:
            if G[v][u] in (1,3):
                idx2 = nodes.index(u) + 1
                print('\\foreach \\x in{' + ','.join(map(str, range(1, s - 1))) + '}{')
                print('  \\pgfmathtruncatemacro{\\xn}{\\x+1}')
                print('  \\draw[pil] (' + mname + '-' + str(idx1) + '-\\x) -- ('\
                    + mname + '-' + str(idx2) + '-\\xn);')
                print('};')


def emacs_vars():
    print('%%% Local Variables:')
    print('%%% mode: latex')
    print('%%% TeX-master: "../master"')
    print('%%% End:')


def dbnprint(G, s, mname='generic', w_gap=0.5, h_gap=0.5, type="obs", stl=''):
    matrix_grid(G, s, mname=mname, w_gap=w_gap, h_gap=h_gap, type=type, stl=stl)
    matrix_edges(G, s, mname=mname)


def getangle(A, B):
    """
    When A and  B are two angles around the clock  returns an angle of
    the line that is connecting them.
    """
    x = array([cos(deg2rad(A)), sin(deg2rad(A))])
    y = array([cos(deg2rad(B)), sin(deg2rad(B))])
    d = y - x
    return rad2deg(math.atan2(d[1], d[0]))


def cdbnprint(G, mtype="obs", bend=5, curve=5, R=1):
    """
    Prints  out  a  compressed  dbn  repesentation  of  the  graph  in
    TikZ/Latex format
    """
    output = StringIO.StringIO()
    BE = set()
    n = len(G)
    nodes = list(G.keys())
    idx = np.argsort([int(v) for v in nodes])
    nodes = [nodes[i] for i in idx]

    g = dict2graph(ecj.cloneBfree(G))
    paintSCC(g, load_data.colors)

    for i in range(0, n):
        node = g.vs[i]['label']
        rc = g.vs[i]["color"]
        print("{ \\definecolor{mycolor}{RGB}{" + str(rc[0]) + "," + str(rc[1]) + "," + str(rc[2]) + "}", file=output)
        mcolor = "fill = {rgb: red," + str(rc[0]) + "; green," + str(rc[1]) +\
            "; blue," + str(rc[2]) + "}"
        print("\\node[" + mtype + ", fill=mycolor] (" + str(node) + ") at (" +\
              str(-i * 360 / n + 180) + ":" + str(R) + ") {" + str(node) + "};}",
              file=output)

#    print >>output,"\\foreach \\name/\\angle in {"+",".join(
#        [nodes[i]+"/"+str(-i*360/n+180) for i in range(0,n)])+"}"
#    print >>output,"\\node["+mtype+"] (\\name) at (\\angle:"\
#        +str(R)+") {\\name};"

    for i in range(0, n):
        v = nodes[i]
        ll = [(v,u) for u in G[v]]
        for l in ll:
            a, b = l
            if G[a][b] in (2,3):
                if not(BE.intersection([(a, b)]) or BE.intersection([(b, a)])):
                    ang_a = -nodes.index(a) * 360 / n + 180
                    ang_b = -nodes.index(b) * 360 / n + 180
                    print('  \\draw[pilip, on layer=back] (' + str(a) + ') -- (' + str(b) + ');', file=output)
            if G[a][b] in (1,3):
                ang_a = -nodes.index(a) * 360 / n + 180
                ang_b = -nodes.index(b) * 360 / n + 180
                if a == b:
                    print("\\path[overlay,draw,pil] (" + str(a) + ")" +\
                        " .. controls +(" + "%.5f" % (bend + ang_a) +\
                        ":" + fstr % (2 * curve) + "mm) and +(" +\
                        "%.5f" % (ang_a - bend) +\
                        ":" + "%.5f" % (2 * curve) + "mm) .. (" + str(b) + ");", file=output)
                else:
                    print("\\path[overlay,draw,pil] (" + str(a) + ")" +\
                        " .. controls +(" + "%.5f" % (bend + getangle(ang_a, ang_b)) +\
                        ":" + fstr % (curve) + "mm) and +(" +\
                        fstr % (getangle(ang_b, ang_a) - bend) +\
                        ":" + fstr % (curve) + "mm) .. (" + str(b) + ");", file=output)
    return output


def gprint(G, mtype="obs", bend=5, curve=5, R=1, layout=None, scale=5):
    """
    Prints out an automatically layout compressed dbn repesentation of
    the graph in TikZ/Latex format
    """
    output = StringIO.StringIO()
    BE = set()
    n = len(G)
    if not layout:
        g = dict2graph(ecj.cloneBfree(G))
        layout = g.layout_fruchterman_reingold(maxiter=50000, coolexp=1.1)
        # layout = g.layout_graphopt(niter=50000, node_charge=0.08)
        layout.center([0, 0])
        layout.scale(float(1 / scipy.absolute(layout.coords).max()))
        layout.scale(R)
        cc = scipy.round_(array(layout.coords), decimals=4)
    else:
        g = dict2graph(ecj.cloneBfree(G))
        cc = array(layout.coords)
    paintSCC(g, load_data.colors)
    for i in range(0, n):
        node = g.vs[i]['label']
        rc = g.vs[i]["color"]
        print("{ \\definecolor{mycolor}{RGB}{"\
            + str(rc[0]) + "," + str(rc[1]) + "," + str(rc[2]) + "}",
              file=output)
        mcolor = "fill = {rgb: red," + str(rc[0]) + "; green," + str(rc[1]) +\
            "; blue," + str(rc[2]) + "}"
        print("\\node[" + mtype + ", fill=mycolor] (" + node + ") at (" +\
            str(cc[i][0]) + "," + str(cc[i][1]) + ") {" + node + "};}",
              file=output)

    for i in range(0, n):
        v = g.vs[i]['label']
        ll = [str(v) + '/' + str(u) for u in G[v]]
        for l in ll:
            a, b = l.split('/')
            if G[a][b] in (2,3):
                if not(BE.intersection([(a, b)]) or BE.intersection([(b, a)])):
                    print('  \\draw[pilip, on layer=back] (' +\
                        a + ') -- (' + b + ');', file=output)
            if G[a][b] in (1,3):
                if a == b:
                    dff = cc[g.vs['label'].index(a)] - scipy.mean(cc, 0)
                    ang = scipy.arctan2(dff[1], dff[0])
                    ang_a = scipy.rad2deg(ang)
                    print("\\path[overlay,draw,pil] (" + a + ")" +\
                        " .. controls +(" + "%.5f" % (bend + ang_a) +\
                        ":" + fstr % (2 * curve) + "mm) and +(" +\
                        "%.5f" % (ang_a - bend) +\
                        ":" + "%.5f" % (2 * curve) + "mm) .. (" + b + ");",
                          file=output)
                else:
                    dff = cc[g.vs['label'].index(b)] \
                        - cc[g.vs['label'].index(a)]
                    ang = scipy.arctan2(dff[1], dff[0])
                    ang_a = scipy.rad2deg(ang)
                    ang_b = ang_a + 180
                    print("\\path[overlay,draw,pil] (" + a + ")" +\
                        " .. controls +(" +\
                        "%.5f" % (bend + ang_a) +\
                        ":" + fstr % (curve) + "mm) and +(" +\
                        fstr % (ang_b - bend) +\
                        ":" + fstr % (curve) + "mm) .. (" + b + ");",
                          file=output)
    return output


def cdbnwrap(G, u, name='AAA', R=1, gap=0.5):
    output = StringIO.StringIO()
    print("\\node[right=" + str(gap) + "cm of " + name + str(u - 1)\
        + ",scale=0.7](" + name + str(u) + "){", file=output)
    print("\\begin{tikzpicture}",  file=output)
    s = cdbnprint(undersample(G, u), mtype='lahid', bend=25, curve=10, R=R)
    print(s.getvalue(),  file=output)
    s.close()
    print("\\end{tikzpicture}",  file=output)
    print("};",  file=output)

    return output


def cdbnsingle(g, scale=0.7, R=1, gap=0.5, mtype="lahid"):
    output = StringIO.StringIO()
    print("\\node[scale=" + str(scale) + "](){", file=output)
    print("\\begin{tikzpicture}", file=output)
    s = cdbnprint(g, mtype=mtype, bend=25, curve=10, R=R)
    print(s.getvalue(), file=output)
    s.close()
    print("\\end{tikzpicture}", file=output)
    print("};", file=output)
    return output


def cdbn_single(G, u, scale=0.7, R=1, gap=0.5, mtype="lahid"):
    return cdbnsingle(undersample(G, u), scale=scale, R=R, gap=gap, mtype=mtype)


def gsingle(g, scale=0.7, R=1, gap=0.5, mtype="lahid", layout=None):
    output = StringIO.StringIO()
    print("\\node[scale=" + str(scale) + "](){", file=output)
    print("\\begin{tikzpicture}", file=output)
    s = gprint(g, mtype=mtype, bend=25, curve=6, R=R, layout=layout)
    print(s.getvalue(), file=output)
    s.close()
    print("\\end{tikzpicture}", file=output)
    print("};", file=output)
    return output


def g_single(G, u, scale=0.7, R=1, gap=0.5, mtype="lahid", layout=None):
    g = undersample(G, u)
    return gsingle(g, scale=scale, R=R, gap=gap, mtype=mtype, layout=layout)


def unfoldplot(G, steps=7, repeats=5, gap=0.5, R=1, hg=0.1, wgap=0.7, name='AAA', stl=''):
    u = 0
    dbnprint(undersample(G, u), repeats, w_gap=wgap,
             h_gap=hg, mname=name + str(u), type='hid', stl=stl)
    print("\\node[left=" + str(gap) + "cm of " + name + str(u) + ",scale=0.7] (C) {")
    print("\\begin{tikzpicture}")
    cdbnprint(G, mtype='hid', bend=15, curve=5, R=R)
    print("\\end{tikzpicture}")
    print("};")
    for u in range(1, steps):
        dbnprint(undersample(G, u), repeats, w_gap=wgap, h_gap=hg, mname=name +
                 str(u), type='ahid', stl=', below=0.25cm of ' + name + str(u - 1))

        print("\\node[left=" + str(gap) + "cm of " + name + str(u) + ",scale=0.7] () {")
        print("\\begin{tikzpicture}")
        cdbnprint(undersample(G, u), mtype='lahid', bend=15, curve=5, R=R)
        print("\\end{tikzpicture}")
        print("};")
    emacs_vars()


def foldplot(G, steps=7, gap=0.5, R=1, hg=0.1, name='AAA', stl='', bend=50, curve=10):
    u = 0
    print('\\node[scale=0.7' + stl + '] (' + name + str(u) + ') {')
    print("\\begin{tikzpicture}")
    s = cdbnprint(G, mtype='hid', bend=bend, curve=curve, R=R)
    print(s.getvalue())
    s.close()
    print("\\end{tikzpicture}")
    print("};")
    for u in range(1, steps):
        s = cdbnwrap(G, u, R=R, gap = gap, name=name)
        print(s.getvalue())
        s.close()
    emacs_vars()


def matrix_fold(G, m, n, R=1, mname='g', w_gap=0.5, h_gap=0.5, stl=''):
    matrix_start(mname=mname, w_gap=w_gap, h_gap=h_gap, stl=stl)
    for i in range(0, n):
        S = []
        for j in range(0, m):
            u = m * i + j
            if u == 0:
                s = cdbn_single(G, u, R=R, mtype="hid")
            else:
                s = cdbn_single(G, u, R=R)
            S.append(s.getvalue())
            s.close()
        print(" ", "& ".join(S) + "\\\\")
    matrix_end()


def gmatrix_fold(G, m, n, R=1, mname='g', w_gap=0.5, h_gap=0.5, stl='', shift=0):
    # g = dict2graph(ecj.cloneBfree(G))
    # layout = g.layout_graphopt(niter=50000, node_charge=0.02)
    # layout.center([0,0])
    # layout.scale(0.01)
    matrix_start(mname=mname, w_gap=w_gap, h_gap=h_gap, stl=stl)
    for i in range(0, n):
        S = []
        for j in range(0, m):
            u = m * i + j + shift
            s = g_single(G, u, R=R, mtype="hid")
            S.append(s.getvalue())
            s.close()
        print(" ", "& ".join(S) + "\\\\")
    matrix_end()


def addselfedge(G, v):
    G[v].update({v: 1})


def gmatrix_list(l, m, n, R=1, mname='g', w_gap=0.5, h_gap=0.5, stl='', shift=0):
    """
    Given a list of graphs prints them out as latex laying out each graph in a force-based layout
    """
    matrix_start(mname=mname, w_gap=w_gap, h_gap=h_gap, stl=stl)
    for i in range(0, n):
        S = []
        for j in range(0, m):
            u = m * i + j + shift
            if u > len(l) - 1:
                break
            s = gsingle(l[u], R=R, mtype="hid")
            S.append(s.getvalue())
            s.close()
        print(" ", "& ".join(S) + "\\\\")
    matrix_end()


def matrix_list(l, m, n, R=1, mname='g', w_gap=0.5, h_gap=0.5, stl='', shift=0):
    """
    Given a list of graphs prints them out as latex laying out each in a circle layout
    """
    matrix_start(mname=mname, w_gap=w_gap, h_gap=h_gap, stl=stl)
    for i in range(0, n):
        S = []
        for j in range(0, m):
            u = m * i + j + shift
            if u > len(l) - 1:
                break
            s = cdbnsingle(l[u], R=R, mtype="hid")
            S.append(s.getvalue())
            s.close()
        print(" ", "& ".join(S) + "\\\\")
    matrix_end()


class WritableObject:

    def __init__(self):
        self.content = []

    def write(self, string):
        self.content.append(string)


def output_graph_figure(g):
    """
        Given a graph will out put a latex file of the graph visualization to figures directory
    """
    sys.stdout = open("../figures/shipfig_figure.tex", "w")
    foldplot(g, steps=1, gap=5, R=15)
    sys.stdout.close()
    sys.stdout = sys.__stdout__