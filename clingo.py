""" This module contains clingo interaction functions """
from __future__ import print_function

import subprocess
import os
from conversions import msl_jclingo2g, rasl_jclingo2g
import json
from calc_procs import get_process_count
CLINGOPATH=''
CAPSIZE=1000
CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))

def clingo_high_version(cpath=CLINGOPATH):
    v = os.popen(cpath+"clingo --version").read()[15:20]
    #v = subprocess.check_output(['clingo', '--version'])[15:20]
    return int(v.split('.')[1]) >= 5

def clingo(command, exact=True, convert=msl_jclingo2g, timeout=0, capsize=CAPSIZE, cpath=CLINGOPATH, pnum=None):
    if pnum is None:
        pnum = PNUM
    clg_start = 'clingo -W no-atom-undefined --configuration=tweety '
    clingo_command = cpath+clg_start+'-t '+str(int(pnum))+',split --outf=2 --time-limit='+str(timeout)\
      +' -n '+str(capsize)+' '
    exp_result = 'SATISFIABLE'
    if not exact:
        exp_result = 'OPTIMUM FOUND'
        clingo_command += ' --opt-mode=opt '
    try:
        p = subprocess.Popen(clingo_command.split(),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=True)
    except:
        return {}
    (output, err) = p.communicate(command)

    if not err:
        result = json.loads(output.decode())
    else:
        if not any([x in err for x in [b'*** Warn', b'*** Info']]):
            print(err)
            return {}
        else:
            result = json.loads(output.decode())
    if result['Result'] == exp_result:
        if exact:
            r = {convert(value['Value']) for value in result['Call'][0]['Witnesses']}
        else:
            r = convert(result['Call'][0]['Witnesses'][-1]['Value'])
        return r
    return {}
