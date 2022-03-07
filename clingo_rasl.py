""" This module contains clingo interaction functions """
from __future__ import print_function

from string import Template
from clingo import clingo
from conversions import g2clingo, rate, rasl_jclingo2g,\
     drasl_jclingo2g, msl_jclingo2g, clingo_preamble,\
     numbered_g2clingo, numbered_g2wclingo, encode_sccs

all_u_rasl_program = """
{edge(X,Y)} :- node(X), node(Y).
directed(X,Y,1) :- edge(X,Y).
directed(X,Y,L) :- directed(X,Z,L-1), edge(Z,Y), L <= U, u(U).
bidirected(X,Y,U) :- directed(Z,X,L), directed(Z,Y,L), node(X;Y;Z), X < Y, L < U, u(U).
countdirh(C):- C = #count { hdirected(X, Y): hdirected(X, Y), node(X), node(Y)}.
countbidirh(C):- C = #count { hbidirected(X, Y): hbidirected(X, Y), node(X), node(Y)}.
equald(L):- { directed(X,Y,L): hdirected(X,Y), node(X), node(Y) } == C, countdirh(C),u(L).
equalb(L):- { bidirected(X,Y,L): hbidirected(X,Y), node(X), node(Y) } == C, countbidirh(C),u(L).
equal(L) :- equald(L), equalb(L).
{trueu(L)} :- equal(L).
equaltest(M) :- 1 < {equal(_)}, equal(M).
min(M):- #min {MM:equaltest(MM)}=M, equaltest(_).
repeat(N):- min(M), equal(N), M<N.
:- directed(X, Y, L), not hdirected(X, Y), node(X), node(Y), trueu(L).
:- not directed(X, Y, L) , hdirected(X, Y), trueu(L).
:- bidirected(X, Y, L), not hbidirected(X, Y), node(X), node(Y), X < Y, trueu(L).
:- not bidirected(X, Y, L), hbidirected(X, Y), X < Y, trueu(L).
:- not trueu(_).
:- min(M), trueu(N), M<N.
    """

drasl_program = """
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).

    :- directed(X, Y, L), not hdirected(X, Y, K), node(X;Y), u(L, K).
    :- bidirected(X, Y, L), not hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.
    :- not directed(X, Y, L), hdirected(X, Y, K), node(X;Y), u(L, K).
    :- not bidirected(X, Y, L), hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.

    """

def weighted_drasl_program(directed, bidirected):
    t = Template("""
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).

    :~ not directed(X, Y, L), hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$directed,X,Y]
    :~ not bidirected(X, Y, L), hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$bidirected,X,Y]
    :~ directed(X, Y, L), no_hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$directed,X,Y]
    :~ bidirected(X, Y, L), no_hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$bidirected,X,Y]
    """)
    return t.substitute(directed=directed, bidirected=bidirected)

def rate(u, uname='u'):
    s = "1 {" + uname + "(1.."+str(u)+")} 1."
    return s

def drate(u, gnum):
    s = f"1 {{u(1..{u}, {gnum})}} 1."
    return s

def rasl_command(g, urate=0, all_u=True):
    if not urate: urate = 1+3*len(g)
    command = g2clingo(g) + ' ' + rate(urate) + ' '
    command += '{edge(X,Y)} :- node(X), node(Y). ' + all_u_rasl_program + ' '
    command += "#show edge/2. "
    command += "#show trueu/1. "
    command += "#show min/1."
    command = command.encode().replace(b"\n", b" ")
    return command

def glist2str(g_list, weighted=False, dm=None, bdm=None):
    if dm is None:
        dm = [None]*len(g_list)
    if bdm is None:
        bdm = [None]*len(g_list)
    s = ''
    for count, (g, D, B) in enumerate(zip(g_list, dm, bdm)):
        if weighted:
            s += numbered_g2wclingo(g, count+1, directed_weights_matrix=D, bidirected_weights_matrix=B) + ' '
        else:
            s +=  numbered_g2clingo(g, count+1) + ' '
    return s

def drasl_command(g_list, max_urate=0, weighted=False, scc=False, dm=None, bdm=None, edge_weights=(1,1)):
    assert len({len(g) for g in g_list}) == 1, "Input graphs have variable number of nodes!"

    if not max_urate: max_urate = 1+3*len(g_list[0])
    n = len(g_list)
    command = clingo_preamble(g_list[0])
    if scc: command += encode_sccs(g_list[0])
    command += f"dagl({len(g_list[0])-1}). "
    command += glist2str(g_list, weighted=weighted, dm=dm, bdm=bdm)+ ' ' # generate all graphs
    command += ' '.join([drate(max_urate, i+1) for i in range(n)]) + ' '
    command += weighted_drasl_program(edge_weights[0], edge_weights[1]) if weighted else drasl_program
    command += f":- M = N, {{u(M, 1..{n}); u(N, 1..{n})}} == 2, u(M, _), u(N, _). "
    command += "#show edge1/2. "
    command += "#show u/2."
    command = command.encode().replace(b"\n", b" ")
    return command

def drasl(glist, capsize, timeout=0, urate=0, weighted=False, scc=False,  dm=None, bdm=None, pnum=None, edge_weights=(1,1)):
    """
    Compute all candidate causal time-scale graphs that could have
    generated all undersampled graphs at all possible undersampling
    rates up to `urate` in `glist` each at an unknown undersampling
    rate.

    :parameter glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (`gunfolds` graphs)

    :parameter capsize: maximum number of candidates to return
    :type capsize: integer

    :parameter timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :parameter urate: maximum undersampling rate to consider
    :type urate: integer

    :parameter weighted: whether the input graphs are weighted or
        imprecize.  If `True` but no weight matrices are provided -
        all weights are set to `1`
    :type weighted: boolean

    :parameter scc: whether to assume that each SCC in the input graph is
        either a singleton or have `gcd=1`.  If `True` a much more
        efficient algorithm is employed.
    :type scc: boolean

    :parameter dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :parameter bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :parameter edge_weights: a tuple of 2 values, the first is importance of matching directed weights when solving optimization problem and the second is for bidirected.
    
    :type edge_weights: tuple with 2 elements
    
    """
    if not isinstance(glist, list):
        glist = [glist]
    return clingo(drasl_command(glist, max_urate=urate, weighted=weighted,
                                scc=scc, dm=dm, bdm=bdm, edge_weights=edge_weights),
                  capsize=capsize, convert=drasl_jclingo2g,
                  timeout=timeout, exact=not weighted, pnum=pnum)

def rasl(g, capsize, timeout=0, urate=0, pnum=None):
    return clingo(rasl_command(g, urate=urate), capsize=capsize, convert=rasl_jclingo2g, timeout=timeout, pnum=pnum)
