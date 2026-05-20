from modules.individual import Individual
from modules.evo_alg.mapElites import algorithm
from modules.simulation.sim1 import sim
import numpy as np
import tomllib

save_path = 'data'

if __name__=='__main__':
    alg = algorithm(resolution=50)
    simulator = sim
    settings  = {'limit': 10.0, 'length': 10.0}

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    population = config['trainer']['population']

    # load checkpoint
    # create Mp Pool
    # generate simulation seed pool
    seeds = np.random.randint(0, 100, size=5)
    run = True
    while run:
        # propose networks          -> returns drone individuals + stats
        individuals, propose_stats = alg.propose(population, None)
        print(propose_stats)

        # score individuals         -> updates individuals + returns stats
        seed = np.random.choice(seeds)
        sim_stats = simulator(individuals, settings, seed=seed)
        print(sim_stats)

        # send results to algorithm -> returns stats
        update_stats = alg.update(individuals)
        print(update_stats)

        # pool transition branch:
        if update_stats['archive_best'] / settings['limit'] > 0.8:
            # update curriculum at pool end
            settings['length'] *= 1.05
            # regenerate seed pool
            seeds = np.random.randint(0, 100, size=5)
            # revalidate existing best drones at pool end
            elites = alg.archive.pop()
            simulator(elites, settings, seed=seeds[0])
            alg.reset(elites)

        # emit stats to logging
        # save checkpoint at generation threshold/new best
        break
