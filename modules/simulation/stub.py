from modules.individual import Individual
from random import randint, random

def sim(individuals: list[Individual], settings):
    for i in individuals:
        i.fitness = randint(0, 99)
        i.descriptors = (randint(0,49), randint(50, 99))
        i.stats = [('action', 'state', 'reward')]

    stats = {'sim': 'success'}
    return stats
