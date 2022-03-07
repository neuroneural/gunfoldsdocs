import clingo as clg
import graphkit as gk
from string import digits
from progressbar import ProgressBar, Percentage
import itertools
from conversions import num2CG, g2vec, vec2g, g2ian
import operator as op
import copy
import sys

from random import choice
from string import digits

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

def num_nATstep(g, step):
    n = len(g)
    b = n*n + ncr(n,2)
    return ncr(b, step)

def num_neighbors(g, step):
    n = len(g)
    b = n*n + ncr(n,2)
    l = 0
    for i in range(step+1):
        l += ncr(b, i)
    return l

def hamming_neighbors(v,step):
    """
    Returns an iterator over all neighbors of the binary vector `v` Hamming `step` away from it
    Arguments:
    - `v`: a binary vector representing a G^u graph
    - `step`: Hamming distance from `v` of the vectors to generate
    """
    for e in itertools.combinations(range(len(v)),step):
        b = copy.deepcopy(v)
        for i in e:
            b[i] = int(not b[i])
        yield b


def find_nearest_reachable(g2, maxsolutions=100,
                           max_depth=5, timeout=3600,
                           cpath='',
                           verbose=True):
    c = 0
    gid = ''.join(choice(digits) for i in range(12))
    s = clg.eqclass(g2, capsize=maxsolutions, timeout = timeout, cpath=cpath)
    if s:
        return s,c
    c += 1
    step = 1
    n = len(g2)
    v = g2vec(g2)
    while True:
        c = 0
        if verbose:
            w = ['neighbors checked @ step ' + str(step) + ': ', Percentage(), ' ']
            pbar = ProgressBar(maxval=num_nATstep(g2,step), widgets=w).start()

        for e in hamming_neighbors(v, step):
            g = vec2g(e, n)
            if not gk.scc_unreachable(g):
                s = clg.eqclass(g,
                                capsize=maxsolutions,
                                timeout=timeout,
                                cpath=cpath)
            else:
                s = set()
            if s:
                if verbose:
                    pbar.finish()
                return s, num_neighbors(g2, step-1) + c

            if verbose:
                pbar.update(c)
                sys.stdout.flush()

            c += 1
        if verbose:
            pbar.finish()

        if step >= max_depth:
            return set(), num_neighbors(g2, step-1) + c
        step += 1
