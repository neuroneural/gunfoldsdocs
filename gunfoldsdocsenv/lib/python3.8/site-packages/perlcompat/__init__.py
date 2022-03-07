#!/usr/bin/env python3
#
# Perl-like utility functions such as warn, die, getopt, and require.
# Copyright (c) 2018-2019, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: perl.py,v 1.10 2019/07/02 02:41:16 ohsaki Exp $
#

import getopt
import sys

def warn(astr):
    """Display warning message ASTR to the standard error output."""
    print(astr, file=sys.stderr)

def die(astr):
    """Display message ASTR to the standard error output and terminate the
program execution."""
    raise SystemExit(astr)

class _Options:
    def set(self, name, val):
        setattr(self, name, val)

def getopts(spec):
    """Parse UNIX-style command line options.  Options are specified by SPEC.
    Parsed options are returned as an object.  A value for option X is
    accessible trhough the object attribute X."""
    # initialize all options with None
    opt = _Options()
    for name in spec:
        if name != ':':
            opt.set(name, None)

    # call getopt in the Python standard library
    try:
        opts, args = getopt.getopt(sys.argv[1:], spec)
    except getopt.GetoptError as e:
        print(e)
        return None

    # save parsed options as object attributes
    for key, val in opts:
        name = key[1:]
        if getopt.short_has_arg(name, spec):
            opt.set(name, val)
        else:
            opt.set(name, True)

    # discard already parsed arguments
    sys.argv[1:] = args
    return opt

def require(version):
    """Abort the program if the current Python interepter does not satisfy
    version requirement (i.e., the version is older than VERSION)."""
    if '.' in version:
        req_major, req_minor = map(int, version.split('.'))
    else:
        req_major, req_minor = int(version), 0
    if sys.version_info < (req_major, req_minor):
        major, minor, micro, release, seerial = sys.version_info
        die("Python {}.{} required--this is only {}.{}, stopped.".format(
            req_major, req_minor, major, minor))

def main():
    warn('test of %s function'.format('warning'))
    opt = getopts('al:') or die('usage: {} [-a] [-l line] [file...]'.format(
        sys.argv[0]))
    print('opt_a = {}'.format(opt.a))
    print('opt_l = {}'.format(opt.l))
    print('args = {}'.format(sys.argv[1:]))
    require('3')
    require('3.5')
    require('3.9')

if __name__ == "__main__":
    main()
