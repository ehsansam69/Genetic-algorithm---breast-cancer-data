from typing import List
from random import choices
from collections import namedtuple
from collections.abc import Callable

Genome = List[int]
population = List[Genome]
FitnessFunc = Callable[[Genome],int]
Thing = namedtuple('Thing', ['name','value','weight'])

things = [
    Thing('Labtop',500,2200),
    Thing('Headphone',150,160),
    Thing('Coffe Mug',60,350),
    Thing('Notepad',40,33),
    Thing('Water bottle',30,192),
]

more_things = [
    Thing('mint',20,10),
    Thing('phone',500,1000),
    Thing('baseballcap',150,150),
    Thing('Socks',10,38),
    Thing('Tissues',15,80)
]+ things

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
# print(fitness(genome=generate_genome(10),things= more_things, weight_limit=2000)) ,,, ezample




#selection function to select the solution to next generation
def selection_pair(population: population, fitness_func : FitnessFunc) -> population:
    return choices(
        population = population,
        weights=[fitness_func(genome) for genom in population],
        k=2
    )

#Cross over function 