"""Bounded background pipeline, with complete stdout/stderr logging and hard stop."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    stages=[['-m','pytest','tests/test_experiment7.py','-q']]
    for arm in ('sugarcrepe','flickr30k'): stages.append(['-m','experiment7.screen','--arm',arm])
    for arm in ('sugarcrepe','flickr30k'):
        stages += [['-m','experiment7.run','--arm',arm,'--stage','pilot'],['-m','experiment7.verify',f'results/experiment_7/{arm}/pilot']]
    for arm in ('sugarcrepe','flickr30k'):
        stages += [['-m','experiment7.run','--arm',arm,'--stage','full'],['-m','experiment7.verify',f'results/experiment_7/{arm}/full']]
    stages.append(['-m','experiment7.report'])
    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
    with Path('experiment7_run.log').open('a',encoding='utf-8',buffering=1) as log:
        for stage in stages:
            # Resume only completed stages; never skip a partially computed run.
            if stage[1]=='experiment7.screen':
                arm=stage[-1]
                if Path(f'data/experiment7_screening/{arm}/screening_artifact_hashes.json').exists():
                    log.write(f'Reusing completed fresh B/16 screening: {arm}\n'); continue
            log.write(f'{datetime.now(timezone.utc).isoformat()} Starting: {" ".join(stage)}\n'); log.flush()
            result=subprocess.run([sys.executable,'-u',*stage],stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode:
                log.write(f'STOPPED exit{result.returncode}; prior evidence preserved\n'); return
        log.write('COMPLETE: results/experiment_7/report.md; STOP ALL EXPERIMENTATION\n')


if __name__=='__main__': main()
