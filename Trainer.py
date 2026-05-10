from modules.individual import Individual
from modules.evo_alg.stub import evostub
from modules.simulation.stub import sim
import tomli

if __name__=='__main__':
    algorithm = evostub()
    simulator = sim

    with open('config.toml', 'rb') as f:
        config = tomli.load(f)
    population = config['trainer']['population']
    
    # load checkpoint
    # create Mp Pool
    # generate simulation seed pool
    run = True
    while run:

        # propose networks          -> returns drone individuals + stats
        individuals, stats = algorithm.propose(population, None)
        print(stats)

        # score individuals         -> updates individuals + returns stats
        stats = sim(individuals, None)
        print(stats)

        # send results to algorithm -> returns stats
        stats = algorithm.update(individuals)
        print(stats)

        # regenerate seed pool
        # pool transition branch:
            # update curriculum at pool end
            # revalidate existing best drones at pool end

        # emit stats to logging
        # save checkpoint at generation threshold/new best
        break

