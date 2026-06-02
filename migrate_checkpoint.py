"""One-off migration: old single-`cma` checkpoint -> split cma_fit / cma_improv.

The old checkpoint pickled the whole `algorithm` object, including:
  - `alg.cma`            : a single CMA arm instance (class was named `cma`)
  - `alg.arms['cma']`    : bound method `alg.cma.ask`
  - `alg.bandit.arms`    : MAB stats keyed by 'cma'

The new code expects `alg.cma_fit` / `alg.cma_improv`, arm keys
'cma_fit'/'cma_improv', and matching bandit keys. `__init__` does NOT run on
unpickle, so we rebuild that scaffolding here and re-save in place.

Requires the `cma = cma_fit` alias in arms.py so the pickle can load at all.
Run once:  python migrate_checkpoint.py
"""
import os
import shutil

from modules.evo_alg.mapElites import MAB, load, save, algorithm
from modules.evo_alg import arms

SAVE_PATH = os.path.join('data', 'MAP_Checkpoint.pkl')
BACKUP    = SAVE_PATH + '.pre_cma_split.bak'


def migrate(alg: algorithm):
    # --- CMA instances -------------------------------------------------------
    # the old `alg.cma` instance is, via the alias, already a cma_fit and its
    # class-level tag now resolves to 'cma_fit' -> reuse it to keep the learned
    # mean/covariance. cma_improv starts fresh (no prior improvement-CMA state).
    old_cma = getattr(alg, 'cma', None)
    alg.cma_fit    = old_cma if isinstance(old_cma, arms.cma_fit) else arms.cma_fit()
    alg.cma_improv = arms.cma_improv()
    if hasattr(alg, 'cma'):
        del alg.cma

    # --- arms dict -----------------------------------------------------------
    alg.arms = {
        'random'     : arms.random,
        'gaussian'   : arms.gaussian,
        'iso'        : arms.iso,
        'cma_fit'    : alg.cma_fit.ask,
        'cma_improv' : alg.cma_improv.ask,
    }

    # --- bandit --------------------------------------------------------------
    # carry over stats for arms that still exist; map old 'cma' stats onto
    # 'cma_fit' (it inherits that arm's learned state). 'cma_improv' keeps the
    # fresh default (value=inf) so UCB1 explores it before trusting it.
    old = alg.bandit
    new = MAB(list(alg.arms.keys()))
    new.total_pulls = old.total_pulls
    new.decay       = old.decay
    for k in new.arms:
        src = k
        if k == 'cma_fit' and 'cma' in old.arms and 'cma_fit' not in old.arms:
            src = 'cma'
        if src in old.arms:
            new.arms[k] = dict(old.arms[src])
    alg.bandit = new

    return alg


if __name__ == '__main__':
    if not os.path.isfile(SAVE_PATH):
        raise SystemExit(f'no checkpoint at {SAVE_PATH}')

    shutil.copy2(SAVE_PATH, BACKUP)
    print(f'backed up -> {BACKUP}')

    alg, settings, seed = load(SAVE_PATH)
    print(f'loaded gen={alg.gen}; old arms={list(alg.arms.keys())}')

    alg = migrate(alg)
    print(f'migrated;   new arms={list(alg.arms.keys())}')
    print(f'bandit arms={list(alg.bandit.arms.keys())}')

    save(SAVE_PATH, alg, settings, seed)
    print(f'saved -> {SAVE_PATH}')
