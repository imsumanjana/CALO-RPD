"""Process-based parallel execution of independent runs with failure isolation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from .seed_manager import SeedManager
from .experiment_runner import run_single,failed_run_from_exception
def _worker(args):config,algo,ri,seeds=args;return run_single(config,algo,ri,seeds)
def run_parallel_resilient(config):
    config.validate();seeds=SeedManager(config.master_seed).generate(config.runs);jobs=[(config,a,ri,seeds[ri]) for ri in range(config.runs) for a in config.algorithms];done=[];failed=[]
    with ProcessPoolExecutor(max_workers=max(1,config.parallel_workers)) as ex:
        fut={ex.submit(_worker,j):j for j in jobs}
        for f in as_completed(fut):
            config0,algo,ri,s=fut[f]
            try:done.append(f.result())
            except Exception as exc:failed.append(failed_run_from_exception(algo,ri,s,exc))
    done.sort(key=lambda x:(x.run_index,config.algorithms.index(x.algorithm)));failed.sort(key=lambda x:(x.run_index,config.algorithms.index(x.algorithm) if x.algorithm in config.algorithms else 999));return done,failed
