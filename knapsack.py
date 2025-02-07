import csv
import numpy as np
import random
from deap import algorithms, base, creator, tools
from collections import defaultdict

# === DEAP/Knapsack Setup ===

# Avoid repeated creation errors if run interactively
try:
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
except RuntimeError:
    pass

try:
    creator.create("Individual", set, fitness=creator.Fitness)
except RuntimeError:
    pass

toolbox = base.Toolbox()

# Constants
IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20

def create_items(seed):
    """
    Create the item dictionary based on the given seed.
    Ensures each seed changes item generation as well.
    """
    random.seed(seed)
    return {
        i: (random.randint(1, 10), random.uniform(0, 100))
        for i in range(NBR_ITEMS)
    }

def evalKnapsack(individual, items):
    """Evaluate the weight and value of the selected items."""
    weight = sum(items[item][0] for item in individual)
    value = sum(items[item][1] for item in individual)
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        # Overweight or too many items => penalize
        return (10000, 0)
    return (weight, value)

def cxSet(ind1, ind2):
    """Crossover: intersection and symmetric difference."""
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2

def mutSet(individual):
    """Mutation: remove or add an element with 50/50 chance."""
    if random.random() < 0.5:
        if individual:  
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return (individual,)

# === Main EA Runner for One Combination & Seed ===
def run_ea(cxpb, mutpb, seed):
    """
    Runs one instance of the evolutionary algorithm.
    Returns (earliest_generation, max_value) for that run.
    """
    # Seed everything
    random.seed(seed)

    # Build items for this seed
    items = create_items(seed)

    # Register to toolbox
    toolbox.register("attr_item", random.randrange, NBR_ITEMS)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate with the 'items' from this seed
    toolbox.register("evaluate", lambda ind: evalKnapsack(ind, items))
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    # EA parameters
    NGEN, MU, LAMBDA = 100, 50, 100

    # Create population
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run EA
    logbook = algorithms.eaMuPlusLambda(
        population=pop, toolbox=toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=cxpb, mutpb=mutpb, ngen=NGEN,
        stats=stats, halloffame=hof, verbose=False
    )

    # logbook[1] has the statistics
    records = logbook[1]

    # Find overall max value across all generations in this run
    all_max_values = [entry["max"][1] for entry in records]
    overall_max_value = max(all_max_values)

    # Earliest generation that achieved overall_max_value
    earliest_generation = None
    for entry in records:
        if entry["max"][1] == overall_max_value:
            earliest_generation = entry["gen"]
            break

    return earliest_generation, overall_max_value


def main():
    # We'll store one CSV of all seeds & combinations:
    summary_filename = "deap_knapsack_summary.csv"

    # We want increments of 0.1 from 0.0 to 1.0
    # We'll keep them as nice floats using round(...)
    results = []
    for seed in range(1, 30):
        for i in range(11):
            cxpb = round(i / 10, 1)
            mutpb = round(1 - (i / 10), 1)

            earliest_gen, max_val = run_ea(cxpb, mutpb, seed)

            # Store in memory
            results.append([
                seed, cxpb, mutpb,
                earliest_gen,  # earliest generation for that run
                max_val        # overall max value for that run
            ])

    # --- Write single summary CSV ---
    with open(summary_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "cxpb", "mutpb",
                         "earliest_gen_of_max_value", "max_value"])
        writer.writerows(results)

    print(f"Summary of all runs saved to {summary_filename}")

    # --- Now Examine the CSV (or the 'results' in memory) ---
    # We'll find the best (cxpb, mutpb) by:
    #   1) Group by (cxpb, mutpb) over seeds
    #   2) Compute average earliest_gen and average max_value
    #   3) Best = highest average max_value; tie-break by earliest average_gen
    combos_dict = defaultdict(list)
    for row in results:
        seed, cxpb, mutpb, earliest_gen, max_val = row
        combos_dict[(cxpb, mutpb)].append((earliest_gen, max_val))

    best_combo = None
    best_combo_avg_max_val = float("-inf")
    best_combo_avg_gen = None

    for combo, runs_list in combos_dict.items():
        # runs_list is a list of (earliest_gen, max_val) for each seed
        avg_gen = np.mean([x[0] for x in runs_list])
        avg_val = np.mean([x[1] for x in runs_list])

        # We want to pick the combo that has the highest avg_val
        # if there's a tie, pick the smallest avg_gen
        if avg_val > best_combo_avg_max_val:
            best_combo = combo
            best_combo_avg_max_val = avg_val
            best_combo_avg_gen = avg_gen
        elif abs(avg_val - best_combo_avg_max_val) < 1e-9:
            # tie in average max_value => check average generation
            if avg_gen < best_combo_avg_gen:
                best_combo = combo
                best_combo_avg_max_val = avg_val
                best_combo_avg_gen = avg_gen

    # best_combo is (cxpb, mutpb)
    print("\n=== BEST COMBINATION ACROSS ALL SEEDS ===")
    print(f"cxpb = {best_combo[0]:.1f}, mutpb = {best_combo[1]:.1f}")
    print(f"Average Max Value = {best_combo_avg_max_val:.4f}")
    print(f"Average Earliest Generation of Max Value = {best_combo_avg_gen:.2f}")

    return best_combo

if __name__ == "__main__":
    main()
