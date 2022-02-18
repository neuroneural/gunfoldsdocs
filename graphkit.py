# tools to construct (random) graphs
from conversions import nx2graph, nxbp2graph, graph2adj, graph2nx
#from gunfolds.tools import ecj
from itertools import combinations
import networkx as nx
from networkx.utils.random_sequence import powerlaw_sequence
from networkx.algorithms.dag import topological_sort
import numpy as np
import random
from numpy.random import randint

def edgelist(g):  # directed
    '''
    return a list of tuples for edges of g
    '''
    l = []
    for n in g:
        l.extend([(n, e) for e in g[n] if g[n][e] in (1, 3)])
    return l


def inedgelist(g):  # missing directed iterator
    '''
    iterate over the list of tuples for edges of g
    '''
    n = len(g)
    for v in g:
        for i in range(1, n + 1):
            if not i in g[v]:
                yield (v, i)
            elif g[v][i] not in (1, 3):
                yield (v, i)


def inbedgelist(g):  # missing bidirected iterator
    '''
    iterate over the list of tuples for edges of g
    '''
    for v in g:
        for w in g:
            if v != w:
                if not w in g[v]:
                    yield (v, w)
                elif g[v][w] not in (2, 3):
                    yield (v, w)


def bedgelist(g):
    """ bidirected edge list with flips """
    l = []
    for n in g:
        l.extend([tuple(sorted((n, e))) for e in g[n] if g[n][e] in (2, 3)])
    l = list(set(l))
    l = l + list(map(lambda x: (x[1], x[0]), l))
    return l


def superclique(n):
    """ All possible edges """
    g = {}
    for i in range(n):
        g[i + 1] = {j + 1: 3 for j in range(n) if j != i}
        g[i + 1][i + 1] = 1
    return g


def complement(G):
    """ return the complement of G """
    n = len(G)
    sq = superclique(n)
    for v in G:
        for w in G[v]:
            sq[v][w] = sq[v][w] - G[v][w]
            if sq[v][w] == 0:
                del sq[v][w]
    return sq


def gtranspose(G):
    """ Transpose (rev. edges of) G """
    GT = {u: {} for u in G}
    for u in G:
        for v in G[u]:
            if G[u][v] in (1,3):
                GT[v][u] = 1        # Add all reverse edges
    return GT


def scale_free(n, alpha=0.7, beta=0.25,
               delta_in=0.2, delta_out=0.2):
    g = nx.scale_free_graph(n, alpha=alpha,
                            beta=beta,
                            delta_in=delta_in, delta_out=delta_out)
    g = nx2graph(g)
    g = gtranspose(g)
    addAring(g)
    return g


def randH(n, d1, d2):
    """ Generate a random H with n nodes """
    g = ringmore(n, d1)
    pairs = [x for x in combinations(g.keys(), 2)]
    for p in np.random.permutation(pairs)[:d2]:
        g[p[0]][p[1]] = g[p[0]].get(p[1], 0) + 2
        g[p[1]][p[0]] = g[p[1]].get(p[0], 0) + 2
    return g


def ring(n):
    g = {}
    for i in range(1, n):
        g[i] = {i + 1: 1}
    g[n] = {1: 1}
    return g


def addAring(g):
    """Add a ring to g in place"""
    for i in range(1, len(g)):
        if g[i].get(i + 1) == 2:
            g[i][i + 1] = 3
        else:
            g[i][i + 1] = 1
    if g[i].get(1) == 2:
        g[i][1] = 3
    else:
        g[i][1] = 1


def upairs(n, k):
    '''n unique nonsequential pairs
    '''
    s = set()
    for p in randint(n, size=(3 * k, 2)):
        if p[1] - p[0] == 1:
            continue
        s.add(tuple(p))
    return list(s)[:k]


def ringarcs(g, n):
    for edge in upairs(len(g), n):
        g[edge[0] + 1][edge[1] + 1] = 1
    return g


def ringmore(n, m):
    return ringarcs(ring(n), m)


def digonly(H):
    """returns a subgraph of H contatining all directed edges of H

    Arguments:

    - `H`: undersampled graph

    """

    g = {n: {} for n in H}
    for v in g:
        g[v] = {w: 1 for w in H[v] if not H[v][w] == 2}
    return g


def _OCE(g1, g2):
    '''
    omission/commision error of g1 referenced to g2
    '''
    s1 = set(edgelist(g1))
    s2 = set(edgelist(g2))
    omitted = len(s2 - s1)
    comitted = len(s1 - s2)

    s1 = set(bedgelist(g1))
    s2 = set(bedgelist(g2))
    bomitted = len(s2 - s1)
    bcomitted = len(s1 - s2)

    return {'directed': (omitted, comitted),
            'bidirected': (bomitted, bcomitted),
            'total': (omitted+bomitted, comitted+bcomitted)}


def _normed_OCE(g1, g2):
    """Return omission and comission errors for directed and

    bidirected edges.

    Omission error is normalized by the number of edges present
    in the ground truth. Commision error is normalized by the
    number of possible edges minus the number of edges present
    in the ground truth.

    Arguments:

    - `g1`: the graph to check

    - `g2`: the ground truth graph
    """
    def sdiv(x,y):
        if y < 1.:
            return 0.
        return x/y

    n = len(g2)
    gt_DEN = float(len(edgelist(g2)))  # number of d  edges in GT
    gt_BEN = float(len(bedgelist(g2))) # number of bi edges in GT
    DEN = n*n       # all posible directed edges
    BEN = n*(n-1)/2 # all possible bidirected edges
    err = OCE(g1, g2)
    nerr = {'directed': (err['directed'][0]/gt_DEN,
                         sdiv(err['directed'][1],(DEN - gt_DEN))),
            'bidirected': (sdiv(err['bidirected'][0], gt_BEN),
                           sdiv(err['bidirected'][1],(BEN - gt_BEN))),
            'total': ((err['directed'][0]+err['bidirected'][0])/(gt_DEN+gt_BEN),
                      sdiv((err['directed'][1]+err['bidirected'][1]),
                           (DEN+BEN - gt_BEN - gt_DEN)))
            }
    return nerr


def OCE(g1, g2, normalized=False):
    if normalized:
        err = _normed_OCE(g1,g2)
    else:
        err = _OCE(g1,g2)
    return err


def clean_leaf_nodes(g):
    for v in g:
        g[v] = {w: g[v][w] for w in g[v] if g[v][w] > 0}


def cerror(d):
    return d['OCE']['directed'][1] / np.double(len(d['gt']['graph']) ** 2 - len(edgelist(d['gt']['graph'])))


def oerror(d):
    return d['OCE']['directed'][0] / np.double(len(edgelist(d['gt']['graph'])))


def bidirected_no_fork(g):
    be = bedgelist(g)
    T = gtranspose(g)
    for e in be:
        if not set(T[e[0]].keys()) & set(T[e[1]].keys()):
            return True
    return False


def no_parents(g):
    T = gtranspose(g)
    for n in T:
        if not T[n]:
            return True
    return False


def no_children(g):
    for n in g:
        if not g[n]:
            return True
    return False


def scc_unreachable(g):
    if bidirected_no_fork(g):
        return True
    if no_parents(g):
        return True
    if no_children(g):
        return True
    return False

# unlike functions from traversal package these do no checking


def addanedge(g, e):
    g[e[0]][e[1]] = 1


def delanedge(g, e):
    g[e[0]].pop(e[1], None)


def addedges(g, es):
    for e in es:
        addanedge(g, e)


def deledges(g, es):
    for e in es:
        delanedge(g, e)


def isdedgesubset(g2star, g2):
    '''
    check if g2star directed edges are a subset of those of g2
    '''
    for n in g2star:
        for h in g2star[n]:
            if h in g2[n]:
                # if g2star has a directed edge and g2 does not
                if g2star[n][h] in (1,3) and g2[n][h] == 2:
                    return False
            else:
                return False
    return True


def isedgesubset(g2star, g2):
    '''
    check if all g2star edges are a subset of those of g2
    '''
    for n in g2star:
        for h in g2star[n]:
            if h in g2[n]:
                # Everything is a subset of 3 (both edge types)
                if g2[n][h] != 3:
                    # Either they both should have a directed edge, or
                    # both should have a bidirected edge
                    if g2star[n][h] != g2[n][h]:
                        return False
            else:
                return False
    return True

def degree_ring(n, d):
    """Generate a ring graph with `n` nodes and average degree `d`
    """
    g = nx.expected_degree_graph([d-1]*n)
    gn = nx2graph(g)
    addAring(gn)
    return gn


def density(g):
    return len(edgelist(g)) / np.double(len(g) ** 2)


def udensity(g):
    return (len(gedgelist(g))+len(bedgelist(g))/2) / np.double(len(g)**2 + len(g)*(len(g)-1)/2)


def mean_degree_graph(node_num, degree):
    g = nx.fast_gnp_random_graph(node_num, degree/node_num, directed=True)
    g = nx2graph(g)
    return g


def pow_degree_graph(node_num, degree, seed=None):
    while True:
        try:
            sequence = powerlaw_sequence(node_num, exponent=degree)
            g = nx.havel_hakimi_graph([int(x) for x in sequence])
            break
        except nx.NetworkXError:
            continue
    g = nx2graph(g)
    return g


def bp_mean_degree_graph(node_num, degree, seed=None):
    G = nx.bipartite.random_graph(node_num, node_num, degree/node_num, seed=seed)
    g = nxbp2graph(G)
    return g


# this function does not work yet - WIP
def bp_pow_degree_graph(node_num, degree, prob=0.7, seed=None):
    while True:
        try:
            sequence = powerlaw_sequence(node_num, exponent=degree)
            g = nx.bipartite.preferential_attachment_graph([int(x) for x in sequence], prob)
            break
        except nx.NetworkXError:
            continue
    g = nxbp2graph(g)
    return g

def remove_tril_singletons(T):
    """Ensure that the DAG resulting from this matrix will not have

    singleton nodes not connected to anything.

    @param T: lower triangular matrix representing a DAG

    @type T: numpy array
    """
    N = T.shape[0]
    neighbors = T.sum(0) + T.sum(1)
    idx = np.where(neighbors == 0)
    if idx:
        for i in idx[0]:
            v1 = i
            while i == v1:
                v1 = randint(0,N-1)
            if i > v1:
                T[i][v1] = 1
            elif i < v1:
                T[v1][i] = 1
    return T

def randomTRIL(N, degree=5, connected=False):
    """Generate a random triangular matrix

    https://stackoverflow.com/a/56514463
    """
    mat = [[0 for x in range(N)] for y in range(N)]
    for _ in range(N):
        for j in range(degree):
            v1 = randint(0,N-1)
            v2 = randint(0,N-1)
            if v1 > v2:
                mat[v1][v2] = 1
            elif v1 < v2:
                mat[v2][v1] = 1
    mat = np.asarray(mat, dtype=np.uint8)
    if connected:
        mat = remove_tril_singletons(mat)
    return mat


def randomDAG(N, degree=5, connected=True):
    adjacency_matrix = randomTRIL(N, degree=degree,
                                  connected=connected)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    if connected:
        components = [x for x in nx.algorithms.components.weakly_connected_components(gr)]
        if len(components) > 1:
            for component in components[1:]:
                v1 = random.choice(tuple(components[0]))
                v2 = random.choice(tuple(component))
                gr.add_edge(v1, v2)
    assert nx.is_directed_acyclic_graph(gr)
    assert nx.algorithms.components.is_weakly_connected(gr)
    return gr



def shift_labels(g, shift):
    new_g = {}
    for v in g:
        new_g[v+shift] = {}
        for w in g[v]:
            new_g[v+shift][w+shift] = g[v][w]
    return new_g


def shift_list_labels(glist):
    if len(glist) < 2:
        return glist
    components = [glist[0]]
    shift = len(glist[0])
    for i in range(1, len(glist)):
        components.append(shift_labels(glist[i], shift))
        shift += len(glist[i])
    return components


def merge_list(glist):
    g = {}
    for e in glist:
        g.update(e)
    return g


def subgraph(g, nodes):
    """Returns a subgraph of `g` that consists of `nodes` and their
    interconnections.

    @param g: gunfolds graph
    @type g: dictionary

    @param nodes: integer valued nodes to include
    @type nodes: list
    """
    nodes = set(nodes)
    sg = {}
    for node in nodes:
        sg[node] = {x:g[node][x] for x in g[node]}
    return sg

def gcd4scc(SCC):
    # first check if there is at least a single simple loop
    x = np.sum([selfloop(_, SCC) for _ in SCC])
    if x > 0: return 1
    g = graph2nx(SCC)
    return ecj.listgcd([len(x) for x in nx.simple_cycles(g)])


def ensure_gcd1(scc):
    """
    If `scc`'s loop structure is not `gcd=1` pick a random node and
    add a self-loop

    @param scc: a strongly connected component
    @type scc: dictionary (gunfolds graph)
    """
    if gcd4scc(scc) > 1:
        a = random.choice([*scc])
        scc[a][a] = 1
    return scc


def update_graph(g, g2):
    """
    Update `g` with connections (or nodes) from `g2`. Both must be at `u=1`
    """
    for v in g2:
        if v in g:
            g[v].update(g2[v])
        else:
            g[v] = g2[v].copy()
    return g

def merge_graphs(glist):
    """
    Merge a list of graphs at `u=1` into a single new graph `g`
    """
    g = {}
    for subg in glist:
        update_graph(g, subg)
    return g


#this function takes any graph, breaks it into SCCs and make sure each SCC has a gcd of 1
def ensure_graph_gcd1(g):
    G = graph2nx(g)
    x = [ensure_gcd1(subgraph(g, c)) for c in nx.strongly_connected_components(G)]
    return merge_graphs([g]+x)


def gcd1_bp_mean_degree_graph(node_num, degree, seed=None):
    g = bp_mean_degree_graph(node_num, degree, seed)
    return ensure_graph_gcd1(g)


def ring_sccs(num, num_sccs, dens=0.5, degree=3, max_cross_connections=3):
    """Generate a random graph with `num_sccs` SCCs, n-nodes each
    """
    dag = randomDAG(num_sccs, degree=degree)
    while nx.is_empty(dag):
        dag = randomDAG(num_sccs, degree=degree)
    if not nx.is_weakly_connected(dag):
        randTree = nx.random_tree(n=num_sccs, create_using=nx.DiGraph)
        dag = remove_loop(nx.compose(dag, randTree), dag=dag)
    ss = shift_list_labels([ensure_gcd1(ringmore(num, max(int(dens * num**2)-num,1) )) for i in range(num_sccs)])
    for v in dag:
        v_nodes = [x for x in ss[v].keys()]
        for w in dag[v]:
            w_nodes = [x for x in ss[w].keys()]
            for i in range(randint(low=1, high=max_cross_connections+1)):
                a = random.choice(v_nodes)
                b = random.choice(w_nodes)
                ss[v][a][b] = 1
    return merge_list(ss)

def selfloop(n, g):
    return n in g[n]

def remove_loop(G,dag):
    try:
        while True:
            lis = nx.find_cycle(G)
            for edge in lis:
                if edge in list(dag.edges):
                    G.remove_edge(edge[0],edge[1])
                    break
    except nx.exception.NetworkXNoCycle:
        assert nx.is_directed_acyclic_graph(G)
        assert nx.is_weakly_connected(G)
        return G
