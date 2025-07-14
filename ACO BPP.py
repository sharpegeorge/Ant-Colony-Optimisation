import random
import numpy as np
import matplotlib.pyplot as plt


# Define parameters
num_items = 500
num_bins_1 = 10
num_bins_2 = 50
max_fitness_evaluations = 10000
p_values = [10, 100]
evaporation_rates = [0.40, 0.30]

# Select bin based on probability
def select_bin_based_on_probability(probabilities):
    return random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]

# Initialize items with weights
def generate_items(problem_type):
    if problem_type == "BPP1":
        return np.array([i for i in range(1, num_items + 1)])
    elif problem_type == "BPP2":
        return np.array([(i ** 2) / 2 for i in range(1, num_items + 1)])

# Initialize pheromone trails
def initialize_pheromones(num_items, num_bins):
    return np.random.uniform(0, 1, (num_items, num_bins))

# Generate paths for each ant
def generate_ant_path(pheromones, num_bins):
    path = []
    for item_idx in range(len(pheromones)):
        probabilities = pheromones[item_idx] / pheromones[item_idx].sum()
        bin_choice = select_bin_based_on_probability(probabilities)
        path.append(bin_choice)
    return path

# Calculate fitness of an ant's path
def calculate_fitness(path, item_weights, num_bins):
    bins = np.zeros(num_bins)
    for item_idx, bin_idx in enumerate(path):
        bins[bin_idx] += item_weights[item_idx]
    return bins.max() - bins.min()

# Update pheromones based on fitness
def update_pheromones(pheromones, paths, fitnesses):
    for path, fitness in zip(paths, fitnesses):
        pheromone_update = 100 / fitness  # Higher fitness means smaller reward
        for item_idx, bin_idx in enumerate(path):
            pheromones[item_idx][bin_idx] += pheromone_update

# Evaporate pheromones across all paths
def evaporate_pheromones(pheromones, evaporation_rate):
    pheromones *= evaporation_rate

# Main ACO function
def ant_colony_optimization(num_bins, item_weights, num_ants, evaporation_rate):
    pheromones = initialize_pheromones(len(item_weights), num_bins)
    best_fitness = float('inf')
    avgFitnesses = []

    # Iterate through maximum number of evaluations allowed
    maxIterations = max_fitness_evaluations // num_ants
    for iteration in range(maxIterations):
        paths = []
        fitnesses = []

        for _ in range(num_ants):
            path = generate_ant_path(pheromones, num_bins)
            fitness = calculate_fitness(path, item_weights, num_bins)
            paths.append(path)
            fitnesses.append(fitness)
            best_fitness = min(best_fitness, fitness)  # Track best fitness

        # Update and evaporate pheromones
        update_pheromones(pheromones, paths, fitnesses)
        evaporate_pheromones(pheromones, evaporation_rate)

        # Averaging fitness for trial and saving
        fitnessAverage = sum(fitnesses) / len(fitnesses)
        avgFitnesses.append(fitnessAverage)

    return best_fitness, avgFitnesses

# Run all experiments
def run_experiments():
    results = []
    for num_bins, problem_type in [(num_bins_1, "BPP1"), (num_bins_2, "BPP2")]:
        item_weights = generate_items(problem_type)

        # Run trials for each different parameter setting
        for num_ants in p_values:
            for evaporation_rate in evaporation_rates:

                # Storing fitness scores for later analysis
                bestFitnesses = []
                avgFitnessesPerTrial = []

                # Run 5 trials per parameter setting
                for trial in range(5):
                    bestFitness, avgFitnesses = ant_colony_optimization(num_bins, item_weights, num_ants, evaporation_rate)
                    bestFitnesses.append(bestFitness)
                    avgFitnessesPerTrial.append(avgFitnesses)

                # Record results
                results.append({
                    "problem_type": problem_type,
                    "num_bins": num_bins,
                    "num_ants": num_ants,
                    "evaporation_rate": evaporation_rate,
                    "bestFitnesses": bestFitnesses,
                    "avgFitnesses": avgFitnessesPerTrial
                })

    # Print results for analysis
    for result in results:
        print(result)
        avgFitnesses = result["avgFitnesses"]
        p = result["num_ants"]
        e = result["evaporation_rate"]
        problem_type = result["problem_type"]

        for trial in avgFitnesses:
            plt.plot(trial)  # Each inner list is treated as y-values with x-values as their indices

        # Add labels and title
        plt.xlabel("Iterations")
        plt.ylabel("Average Fitness")
        plt.title(f"Average Fitness {problem_type} p={p} e={e}")
        # Add legend
        plt.legend([f"Trial {i+1}" for i in range(len(trial))],  loc="upper right")

        plt.show()


run_experiments()