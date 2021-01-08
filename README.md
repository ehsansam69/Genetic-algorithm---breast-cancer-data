# Genetic-algorithm---breast-cancer-data
Classification method for diagnosis of breast cancer:

Here are quick steps for how the genetic algorithm works:
Initial Population– Initialize the population randomly based on the data.
Fitness function– Find the fitness value of the each of 
the chromosomes(a chromosome is a set of parameters which 
define a proposed solution to the problem that the genetic 
algorithm is trying to solve)
Selection– Select the best fitted chromosomes as parents to pass 
           the genes for the next generation and create a new population
Cross-over– Create new set of chromosome by combining the parents 
            and add them to new population set
Mutation– Perfrom mutation which alters one or more gene values in a 
          chromosome in the new population set generated. Mutation helps in getting more diverse oppourtinity.Obtained population will be used in the next generation
          
In this train code we used sklearn dataset , breast_cancer , for train our data model and predict the label of data. 
