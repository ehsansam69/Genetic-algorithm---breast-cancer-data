from typing import List
from random import choices
from collections import namedtuple

Genome = List[int]
population = List[Genome]
Thing = namedtuple('Thing', ['name','value','weight'])

things = [
    Thing('Labtop',500,2200),
    Thing('Headphone',150,160),
    Thing('Coffe Mug',60,350),
    Thing('Notepad',40,33),
    Thing('Water bottle',30,192),
]

#genetic representation of a solutions
def generate_genome(lenght: int) ->Genome:
    return choices([0,1], k=lenght)

#a function to generate new solution
def generate_population(size:int , genome_lenght:int) -> population:
    return [generate_genome(genome_lenght) for _ in range(size)]

#fitness function to evaluate solutions
# determine the value of a random generated Genome
def fitness(genome: Genome, things : [Thing], weight_limit: int) ->int:
    if len(genome) != len(things):
        raise ValueError('genome and things must be of the same lenght')

    weight = 0
    value = 0

    for i,thing in enumerate(things):
        if genome[i] == 1:
            weight +=thing.weight
            value +=thing.value

            if weight > weight_limit:
                return 0

    return value


print(fitness(genome=generate_genome(5),things= things, weight_limit=20000000))