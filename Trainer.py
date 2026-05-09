from modules.individual import Individual
# algorithm =
# simulator =

if __name__=='__main__':
    # load checkpoint
    # create Mp Pool
    # generate simulation seed pool
    run = True
    while run:

        # propose networks          -> returns drone individuals + stats

        # score individuals         -> updates individuals + returns stats

        # send results to algorithm -> returns stats

        # regenerate seed pool
        # pool transition branch:
            # update curriculum at pool end
            # revalidate existing best drones at pool end

        # emit stats to logging
        # save checkpoint at generation threshold/new best

        pass
