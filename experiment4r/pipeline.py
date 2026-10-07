from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    stages=[['-m','pytest','tests/test_experiment4r.py','-q'],['-m','experiment4r.prepare'],
        ['-m','experiment4r.run','--stage','pilot'],['-m','experiment4r.verify','results/experiment_4R_full_flickr30k/pilot'],
        ['-m','experiment4r.run','--stage','full'],['-m','experiment4r.verify'],['-m','experiment4r.report']]
    with Path('experiment4r_run.log').open('a',encoding='utf-8',buffering=1) as log:
        for stage in stages:
            log.write(f'{datetime.now(timezone.utc).isoformat()} Starting: {" ".join(stage)}\n'); log.flush()
            result=subprocess.run([sys.executable,'-u',*stage],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
            if result.returncode: log.write(f'STOPPED exit{result.returncode}; no replacement or method adjustment\n'); return
        log.write('COMPLETE: results/experiment_4R_full_flickr30k/report.md; STOP\n')


if __name__=='__main__': main()
