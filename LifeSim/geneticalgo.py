import random

# Define the fitness function
def fitness(individual):
    # This function should return the fitness of the individual
    # For simplicity, let's assume the fitness is the value of the individual itself
    return individual

# Define the population size
POPULATION_SIZE = 100

# Define the number of generations
NUM_GENERATIONS = 100

# Create the initial population
population = [random.randint(0, 100) for _ in range(POPULATION_SIZE)]

# Evolve the population
for _ in range(NUM_GENERATIONS):
    # Evaluate the fitness of each individual in the population
    fitnesses = [fitness(individual) for individual in population]

    # Select the parents based on their fitness
    parents = random.choices(population, weights=fitnesses, k=2)

    # Perform crossover to create the offspring
    # Since our individuals are integers, we'll just average the parents
    offspring = (parents[0] + parents[1]) // 2

    # Perform mutation on the offspring
    # We'll add or subtract 1 with a 10% chance
    if random.random() < 0.1:  # 10% chance of mutation
        offspring += random.choice([-1, 1])

    # Replace the least fit individual in the population with the offspring
    min_fitness_index = fitnesses.index(min(fitnesses))
    population[min_fitness_index] = offspring

# Print the best individual in the population
fitnesses = [fitness(individual) for individual in population]
max_fitness_index = fitnesses.index(max(fitnesses))
print(population[max_fitness_index])