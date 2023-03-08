import random

def getRandomSeed():
    random_seed = int(1 / (random.random() + 0.00000001) * 100)
    return random_seed

