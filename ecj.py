#from gunfolds.data.testgraphs import *


def walk(G, s, S=set()):
    P, Q = dict(), set()
    P[s] = None
    Q.add(s)
    while Q:
        u = Q.pop()
        for v in G[u].difference(P, S):
            Q.add(v)
            P[v] = u
    return P


def dfs_topsort(G):
    S, res = set(), []

    def recurse(u):
        if u in S:
            return
        S.add(u)
        for v in G[u]:
            recurse(v)
        res.append(u)
    for u in G:
        recurse(u)
    res.reverse()
    return res


def tr(G):                      # Transpose (rev. edges of) G
    GT = {}
    for u in G:
        GT[u] = set()   # Get all the nodes in there
    for u in G:
        for v in G[u]:
            GT[v].add(u)        # Add all reverse edges
    return GT


def scc(G):                   # Kosaraju's algorithm
    GT = tr(G)                # Get the transposed graph
    sccs, seen = [], set()
    for u in dfs_topsort(G):   # DFS starting points
        if u in seen:
            continue  # Ignore covered nodes
        C = walk(GT, u, seen)  # Don't go "backward" (seen)
        seen.update(C)         # We've now seen C
        sccs.append(C)         # Another SCC found
    return sccs


def cloneBfree(G):
    D = {}
    for v in G:
        D[v] = {}
        for u in G[v]:
            if G[v][u] in (1,3):
                D[v][u] = 1
    return D


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def listgcd_r(l):
    if len(l) > 0:
        return gcd(l[0], listgcd_r(l[1:]))
    else:
        return 0


def listgcd(l):
    mygcd = 0
    if len(l) > 0:
        for x in l:
            mygcd = gcd(x, mygcd)
    return mygcd


def reachable(s, G, g):
    S, Q = set(), []
    Q.append(s)
    while Q:
        u = Q.pop()
        if u in S:
            continue
        if g in G[u]:
            return True
        S.add(u)
        Q.extend(G[u])
    return False


def has_unit_cycle(G, path):
    """ check  if two  unequal  length  paths can  be  compensated by  their
        elementary cycles """
    for v in path:
        if v in G[v]:
            return True
    return False
