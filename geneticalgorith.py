from typing import List, Callable,Tuple
from random import choices,randint,randrange,random
from collections import namedtuple
from functools import partial
import time


Genome = List[int]
population = List[Genome]
FitnessFunc = Callable[[Genome],int]
PopulateFunc = Callable[[], population]
SelectionFunc = Callable[[population, FitnessFunc], Tuple[Genome,Genome]]
CrossoverFunc = Callable[[Genome,Genome],Tuple[Genome,Genome]]
MutationFunc =Callable[[Genome],Genome]

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
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

#Cross over function
def single_pointcrossover(a: Genome, b:Genome)->Tuple[Genome,Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same lenght")

    lenght = len(a)
    if lenght <2:
        return a,b
    p = randint(1, lenght-1)
    return a[0:p]+b[p:], b[0:p]+a[p:]

#mutation function
def mutation(genome: Genome, num: int = 1, probability:float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index]-1)
    return genome

# a def for running evolution
def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func : SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc=single_pointcrossover,
        mutution_func: MutationFunc=mutation,
        generation_limit:int=100,

)->Tuple[population,int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population)/2)-1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0],parents[1])
            offspring_a = mutution_func(offspring_a)
            offspring_b = mutution_func(offspring_b)
            next_generation +=[offspring_a,offspring_b]

        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population,i


#running :
start = time.time()
population,generation = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_lenght=len(more_things)
    ),
    fitness_func=partial(
        fitness,things = more_things, weight_limit = 3000

    ),
    fitness_limit=1310,
    generation_limit=100
)
end = time.time()


def genome_to_things(genome:Genome, things:[Thing]) ->[Thing]:
    result=[]
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result +=[thing.name]
    return result

print(f"number of generations {generation}")
print(f"time: {end-start}s")
print(f"best solution{genome_to_things(population[0],more_things)}")




