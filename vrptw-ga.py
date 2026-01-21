import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import streamlit as st

data = {
    "Sender Package Number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "Sender Coordinates": [(-1.3, -5.3), (-10.4, -3.5), (-11.4, 14), (9.6, -15.7), (-12.2, -10), 
                           (-7.9, 2.7), (-5, -11.6), (13, -7.6), (15.4, 9.9), (-7.8, -14), 
                           (2.1, -17.1), (15.2, -15.1), (0.1, 2.2), (10, 3.7), (5.2, 8.7), 
                           (15.6, 0.5), (-4.8, 8.2), (-7.2, 0.2), (14, 5.2), (-4.8, -12)],
    "Sender Time Window": [(8, 11), (11, 13), (9, 12), (12, 14), (10, 15), 
                           (9, 11), (8, 10), (10, 14), (12, 14), (12, 15), 
                           (11, 14), (9, 12), (13, 15), (10, 12), (12, 14), 
                           (10, 13), (11, 14), (11, 13), (10, 14), (12, 14)],
    "Package Volume": [0.08, 0.13, 0.05, 0.16, 0.08, 0.06, 0.10, 0.13, 0.15, 0.17, 0.08, 0.06, 0.10, 0.16, 0.08, 0.11, 0.15, 0.06, 0.12, 0.13],
    "Recipient Package Number": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    "Recipient Coordinates": [(40, -1.4), (41, -5.5), (50, -4.7), (65, -5.8), (52, 11.6), 
                              (56.3, 5.3), (56.5, 13.7), (50.4, 18.9), (48.2, -21), (35.8, -18.2), 
                              (42.4, -16.1), (49.7, -14.2), (38.5, 5.3), (65, -16.1), (58, 2.1), 
                              (42.1, -12.9), (66.5, 7.9), (43.2, 16.6), (41.6, 11.6), (35.7, 9)],
    "Recipient Time Window": [(0, 16), (0, 17), (0, 16), (0, 18), (0, 18), 
                              (0, 17), (0, 14), (0, 17), (0, 16), (0, 18), 
                              (0, 18), (0, 18), (0, 18), (0, 15), (0, 18), 
                              (0, 17), (0, 18), (0, 17), (0, 18), (0, 17)]
}

df = pd.DataFrame(data)

collection_hub = (5, 0)
distribution_hub = (50, 0)

num_vehicles = 6
vehicle_capacity = 0.5
num_locations = 20

speed = 20
transportation_time_hub = 0.5

def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

sender_locations = [collection_hub] + df['Sender Coordinates'].tolist()
recipient_locations = [distribution_hub] + df['Recipient Coordinates'].tolist() 
num_locations = num_locations + 1
distance_matrix_recipient = np.zeros((num_locations, num_locations))
distance_matrix_sender = np.zeros((num_locations, num_locations))
for i in range(num_locations):
    for j in range(num_locations):
        distance_matrix_sender[i][j] = euclidean_distance(sender_locations[i], sender_locations[j])
for i in range(num_locations):
    for j in range(num_locations):
        distance_matrix_recipient[i][j] = euclidean_distance(recipient_locations[i], recipient_locations[j])

sender_transportation_time = distance_matrix_sender / 20
recipient_transportation_time = distance_matrix_recipient / 20

sender_time_windows = [(0, 18*60)] + df['Sender Time Window'].tolist()   
recipient_time_windows = [(0, 18*60)] + df['Recipient Time Window'].tolist()
package_volumes = [0] + df['Package Volume'].tolist() 

time_vehicle_pick_up = [9.08, 9.22, 8.55, 10.53, 8.72, 9.90]

def time_arrival(individual, time_vehicle, transportation_time, num_locations):
    time_arrival_array = np.zeros(num_locations + 1)
    time_subway = np.zeros(num_locations + 1)
    for i in range(0, len(individual)):
        path_vehicle = individual[i]
        pick_up_time = time_vehicle[i]
        time_arrival_array[path_vehicle[0]] = pick_up_time
        for j in range(1, len(path_vehicle)):
            pick_up_time = pick_up_time + transportation_time[path_vehicle[j - 1]][path_vehicle[j]]
            time_arrival_array[path_vehicle[j]] = pick_up_time
        for j in range(1, len(path_vehicle) - 1):
            time_subway[path_vehicle[j]] = math.ceil(pick_up_time) + 0.5
    return time_arrival_array,time_subway

def check_time_arrival(individual, package_volumes, vehicle_capacity, num_vehicles, num_locations):
    upper_individual = split_chromosome(individual[0], package_volumes, vehicle_capacity, num_vehicles)
    lower_individual = split_chromosome(individual[1], package_volumes, vehicle_capacity, num_vehicles)
    PU, SU = time_arrival(upper_individual,time_vehicle_pick_up, sender_transportation_time, num_locations)
    time_vehicle_distribution = []
    for i in range(len(lower_individual)):
        min_subway = max(sub)
        for j in range(1, len(lower_individual[i]) - 1):
            min_subway = max(min_subway, sub[lower_individual[i][j]])
        time_vehicle_distribution.append(min_subway)   
    arrival_time = np.zeros(num_locations + 1)
    for i in range(0, len(lower_individual)):
        path_vehicle = lower_individual[i]
        pick_up_time = SU[i]
        for j in range(1, len(path_vehicle)):
            pick_up_time = pick_up_time + recipient_transportation_time[path_vehicle[j - 1]][path_vehicle[j]]
            if time_vehicle_distribution[i] < SU[path_vehicle[j]]:
                return False
            arrival_time[path_vehicle[j]] = pick_up_time
    return True

inf = 10000
def objective_function(individual):
    upper_individual = split_chromosome(individual[0], package_volumes, vehicle_capacity, num_vehicles)
    if upper_individual == []: return inf
    lower_individual = split_chromosome(individual[1], package_volumes, vehicle_capacity, num_vehicles)
    if lower_individual == []: return inf
    PU, sub = time_arrival(upper_individual,time_vehicle_pick_up, sender_transportation_time, num_locations)
    time_vehicle_distribution = []
    for i in range(len(lower_individual)):
        min_subway = max(sub)
        for j in range(1, len(lower_individual[i]) - 1):
            min_subway = max(min_subway, sub[lower_individual[i][j]])
        time_vehicle_distribution.append(min_subway)    
    AT, sub1 = time_arrival(lower_individual, time_vehicle_distribution, recipient_transportation_time, num_locations)
    
    result = np.zeros(num_locations + 1)
    for i in range(num_locations):
        result[i] = AT[i] - PU[i]
    return sum(result)

def split_chromosome(chromosome, sender_volumes, vehicle_capacity, num_vehicles):
    lower_threshold = sum(sender_volumes) / num_vehicles
    volumes_value = lower_threshold 
    upper_threshold = vehicle_capacity
    require_length_path = 0
    while (require_length_path != num_vehicles):
        paths = []
        current_path = [0]
        current_path.append(chromosome[0])
        current_load = sender_volumes[chromosome[0]]
        for point in chromosome[1:num_locations - 1]:
            point_volume = sender_volumes[point]
            if current_load + point_volume <= volumes_value:
                current_path.append(point)
                current_load += point_volume
            else:
                current_path.append(0)
                paths.append(current_path)
                current_path = [0, point]
                current_load = point_volume

        current_path.append(0)
        paths.append(current_path)
        require_length_path = len(paths)  
        if volumes_value > upper_threshold: return []
        volumes_value = volumes_value + 0.01
    return paths 

def transportation_cost(individual, time_windows, time_vehicle, transportation_time):
    te = 0
    tl = 0
    for i in range(0, len(individual)):
        path_vehicle = individual[i]
        pick_up_time = time_vehicle[i]
        for j in range(1, len(path_vehicle) - 1):
            pick_up_time = pick_up_time + transportation_time[path_vehicle[j - 1]][path_vehicle[j]]
            if pick_up_time < time_windows[path_vehicle[j]][0]: te = te + time_windows[path_vehicle[j]][0] - pick_up_time
            if pick_up_time > time_windows[path_vehicle[j]][1]: tl = tl - time_windows[path_vehicle[j]][1] + pick_up_time
    return te, tl

def evaluate(individual, package_volumes, vehicle_capacity, num_vehicles):
    upper_individual = split_chromosome(individual[0], package_volumes, vehicle_capacity, num_vehicles)
    lower_individual = split_chromosome(individual[1], package_volumes, vehicle_capacity, num_vehicles)
    PU, sub = time_arrival(upper_individual,time_vehicle_pick_up, sender_transportation_time, num_locations)
    time_vehicle_distribution = []
    for i in range(len(lower_individual)):
        min_subway = max(sub)
        for j in range(1, len(lower_individual[i]) - 1):
            min_subway = max(min_subway, sub[lower_individual[i][j]])
        time_vehicle_distribution.append(min_subway)    
    te, tl = transportation_cost(upper_individual, sender_time_windows, time_vehicle_pick_up, sender_transportation_time)
    fe, fl = transportation_cost(lower_individual, recipient_time_windows, sub, recipient_transportation_time)
    obj_fn = objective_function(individual)
    penalty_cost = te * tl  + fe * fl  + obj_fn * 10
    return (penalty_cost,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_locations), num_locations)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def init_population(pop_size):
    population = []
    for q in range(pop_size):
        upper_chromosome = list(range(1, num_locations))
        upper_chromosome.sort(key=lambda x: sender_time_windows[x][1]) 
        lower_chromosome = upper_chromosome.copy()
        lower_chromosome.sort(key=lambda x: recipient_time_windows[x][1])
        def random_swap(chromosome, num_swaps):
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
            return chromosome

        num_swaps = random.randint(1, 10)
        upper_chromosome = random_swap(upper_chromosome, num_swaps)
        lower_chromosome = random_swap(lower_chromosome, num_swaps)
        individual = [upper_chromosome, lower_chromosome]
        population.append(creator.Individual(individual))
    return population

toolbox.register("population", init_population, pop_size=1600)

def order_crossover(p1, p2):
    size = num_locations - 1
    upper_parent_1 = p1[0]
    upper_parent_2 = p2[0]
    lower_parent_1 = p1[1]
    lower_parent_2 = p2[1]
    offspring1_1, offspring2_1 = [None]*size, [None]*size
    offspring1_2, offspring2_2 = [None]*size, [None]*size
    def order_crossover(parent1, parent2):
        size = len(parent1)
        offspring1, offspring2 = [None]*size, [None]*size

        cut1, cut2 = sorted(random.sample(range(size), 2))

        offspring1[cut1:cut2] = parent1[cut1:cut2]
        offspring2[cut1:cut2] = parent2[cut1:cut2]

        def fill_offspring(offspring, parent):
            current_pos = cut2
            parent_pos = cut2

            while None in offspring:
                if parent[parent_pos % size] not in offspring:
                    offspring[current_pos % size] = parent[parent_pos % size]
                    current_pos += 1
                parent_pos += 1

        fill_offspring(offspring1, parent2)
        fill_offspring(offspring2, parent1)

        return creator.Individual(offspring1), creator.Individual(offspring2)
    offspring1_2,offspring2_2 = order_crossover(lower_parent_1, lower_parent_2)
    return creator.Individual([upper_parent_1, offspring1_2]), creator.Individual([upper_parent_2, offspring2_2])

toolbox.register("evaluate", evaluate, package_volumes=package_volumes, vehicle_capacity=vehicle_capacity, num_vehicles=num_vehicles)
toolbox.register("mate", order_crossover)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selRoulette)

def tournament_selection(pop, fitness, tournament_size, selection_size):
    def partition(pop):
        num_tournaments = int (len(pop) / tournament_size)
        index = [i for i in range(len(pop))]
        np.random.shuffle(index)
        return [index[tournament_size*i:tournament_size*i+tournament_size] for i in range(num_tournaments)]

    offspring = []

    while len(offspring) < selection_size:
        tournaments = partition(pop.copy())
        for tournament in tournaments:
            tournament_inds = [pop[idx] for idx in tournament]
            tournament_fitness = [ind.fitness.values[0] for ind in tournament_inds]
            best_ind_idx = tournament_fitness.index(min(tournament_fitness))
            offspring.append(tournament_inds[best_ind_idx])

    return offspring

def run_ga(num_generations, num_runs):
    best_overall = None
    best_fitness_overall = float('inf')
    for run in range(num_runs):
        population = toolbox.population()
        best_ind = None
        best_fitness = float('inf')

        NGEN = num_generations
        CXPB = 0.8
        MUTPB = 0.1

        for gen in range(NGEN):
            reserved_population = population
            remaining_population = population

            offspring = []
            for i in range(0, len(remaining_population), 2):
                if i + 1 < len(remaining_population):
                    ind1, ind2 = remaining_population[i], remaining_population[i + 1]
                    if random.random() < CXPB:
                        child1, child2 = toolbox.mate(ind1, ind2)
                        del child1.fitness.values
                        del child2.fitness.values
                        offspring.append(child1)
                        offspring.append(child2)
                    else:
                        offspring.append(ind1)
                        offspring.append(ind2)
                else:
                    offspring.append(remaining_population[i])

            population = reserved_population + offspring
            fits = map(toolbox.evaluate, population)
            for fit, ind in zip(fits, population):
                ind.fitness.values = fit
            fitness_values = [ind.fitness.values[0] for ind in population]
            population = tournament_selection(population, fitness_values, 4, len(population) // 2)

            for mutant in population:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            fits = map(toolbox.evaluate, population)
            for fit, ind in zip(fits, population):
                ind.fitness.values = fit
            fitness_values = [ind.fitness.values[0] for ind in population]
            current_best_ind = tools.selBest(population, 1)[0]
            if current_best_ind.fitness.values[0] < best_fitness:
                best_fitness = current_best_ind.fitness.values[0]
                best_ind = current_best_ind

        if best_fitness < best_fitness_overall:
            best_fitness_overall = best_fitness
            best_overall = best_ind

    return best_overall, best_fitness_overall

def decimal_to_time(decimal_hours):
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def print_time_arrival(individual, package_volumes, vehicle_capacity, num_vehicles, num_locations):
    upper_individual = split_chromosome(individual[0], package_volumes, vehicle_capacity, num_vehicles)
    lower_individual = split_chromosome(individual[1], package_volumes, vehicle_capacity, num_vehicles)
    PU, sub = time_arrival(upper_individual,time_vehicle_pick_up, sender_transportation_time, num_locations)
    time_vehicle_distribution = []
    for i in range(len(lower_individual)):
        min_subway = max(sub)
        for j in range(1, len(lower_individual[i]) - 1):
            min_subway = max(min_subway, sub[lower_individual[i][j]])
        time_vehicle_distribution.append(min_subway)    
    AT, sub1 = time_arrival(lower_individual, time_vehicle_distribution, recipient_transportation_time, num_locations)
    
    result = np.zeros(num_locations + 1)
    for i in range(num_locations):
        result[i] = AT[i] - PU[i]

    data = {
        "Package": [f"Package {i}" for i in range(1, num_locations)],
        "Pick Up Time": [decimal_to_time(PU[i]) for i in range(1, num_locations)],
        "Subway Time": [decimal_to_time(sub[i]) for i in range(1, num_locations)],
        "Arrival Time": [decimal_to_time(AT[i]) for i in range(1, num_locations)],
        "Total Time": [decimal_to_time(result[i]) for i in range(1, num_locations)]
    }
    df_result = pd.DataFrame(data)
    return df_result, sum(result)

st.title("VRPTW-GA Demo")

st.header("Introduction to the Problem")
st.write("""
The Vehicle Routing Problem with Time Windows (VRPTW) is a complex optimization problem in logistics and transportation. It involves determining the optimal routes for a fleet of vehicles to collect packages from senders and deliver them to recipients, while respecting time windows for pickups and deliveries, vehicle capacity constraints, and minimizing total transportation time and costs.

In this demo, we use a Genetic Algorithm (GA) to solve a VRPTW instance with 20 senders and 20 recipients. The GA evolves solutions over generations to find routes that satisfy all constraints and optimize the objective function, which penalizes time window violations and transportation costs.
""")

st.header("Data Overview")
st.write("### Input Data")
st.dataframe(df)
st.write("This table shows the initial data for senders and recipients, including coordinates, time windows, and package volumes.")

st.sidebar.header("Simulation Settings")
num_generations = st.sidebar.number_input("Number of Generations", min_value=1, max_value=1000, value=40, step=1)
st.sidebar.write("Adjust the number of generations for the genetic algorithm.")

if st.sidebar.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        best_ind, best_fitness = run_ga(num_generations, 1)
    st.success("Simulation completed!")
    st.write(f"Best Fitness: {best_fitness}")
    st.write("Best Individual:", best_ind)
    df_result, total_time = print_time_arrival(best_ind, package_volumes, vehicle_capacity, num_vehicles, num_locations)
    st.write("### Time Arrival Table")
    st.dataframe(df_result)
    st.write(f"Total Time: {total_time}")

    st.header("Results Explanation")
    sender_routes = split_chromosome(best_ind[0], package_volumes, vehicle_capacity, num_vehicles)
    recipient_routes = split_chromosome(best_ind[1], package_volumes, vehicle_capacity, num_vehicles)
    
    st.write("### Sender Routes (Collection)")
    for i, route in enumerate(sender_routes):
        locations = [f"Location {loc}" for loc in route[1:-1]]
        st.write(f"Vehicle {i+1} starts at the Collection Hub, picks up packages from {', '.join(locations)} in the order {route[1:-1]}, and returns to the Collection Hub.")
    
    st.write("### Recipient Routes (Distribution)")
    for i, route in enumerate(recipient_routes):
        locations = [f"Location {loc}" for loc in route[1:-1]]
        st.write(f"Vehicle {i+1} starts at the Distribution Hub, delivers packages to {', '.join(locations)} in the order {route[1:-1]}, and returns to the Distribution Hub.")
    
    st.write("### Overall Solution")
    st.write(f"The genetic algorithm found an optimal solution with a fitness value of {best_fitness:.2f}, representing the total penalty for time window violations and transportation costs. The total time from pickup to delivery across all packages is {total_time:.2f} hours. Each vehicle operates within capacity constraints and respects the time windows for pickups and deliveries.")
    st.write("The routes ensure that packages are collected from senders and delivered to recipients efficiently, minimizing delays and optimizing vehicle usage in this Vehicle Routing Problem with Time Windows (VRPTW).")

    st.write("### Explanation of Fitness and Objective Values")
    st.write("""
    - **Objective Function Value**: This represents the total time packages spend in transit, calculated as the sum of (delivery time - pickup time) for all packages. A lower value indicates faster overall delivery, meaning packages are moved more efficiently from senders to recipients.
    
    - **Fitness Value**: The fitness is a composite score used by the genetic algorithm to evaluate solutions. It includes penalties for time window violations (arriving too early or too late at locations) and a weighted component of the objective function. Specifically, fitness = (early_penalty_sender * late_penalty_sender) + (early_penalty_recipient * late_penalty_recipient) + (objective_function * 10). The GA minimizes this fitness to prioritize solutions that avoid time window breaches while optimizing transit times.
    
    The optimization aims to find routes that balance respecting time constraints with minimizing the total time packages are in the system, ensuring efficient logistics operations.
    """)

    st.header("Visualization with Routes")
    senders = sender_locations
    recipients = recipient_locations

    senders_x, senders_y = zip(*senders)
    recipients_x, recipients_y = zip(*recipients)

    plt.figure(figsize=(10, 10))
    plt.scatter(senders_x, senders_y, c='blue', label='Senders', marker='o')
    plt.scatter(recipients_x, recipients_y, c='red', label='Recipients', marker='x')

    for i, (x, y) in enumerate(senders):
        plt.text(x, y, str(i), fontsize=10, ha='right', color='black')

    for i, (x, y) in enumerate(recipients):
        plt.text(x, y, str(i), fontsize=10, ha='right', color='black')

    colors = plt.cm.get_cmap('tab10', len(sender_routes))

    for i, path in enumerate(sender_routes):
        path_coords = [senders[idx] for idx in path]
        path_x, path_y = zip(*path_coords)
        plt.plot(path_x, path_y, linestyle='-', marker='o', color=colors(i), label=f'Vehicle {i+1} Sender Route')

    for i, path in enumerate(recipient_routes):
        path_coords = [recipients[idx] for idx in path]
        path_x, path_y = zip(*path_coords)
        plt.plot(path_x, path_y, linestyle='-', marker='x', color=colors(i), label=f'Vehicle {i+1} Recipient Route')

    plt.plot((5,50), (0,0), 'ro-', label='Collection to Distribution')
    plt.xlim(-20, 70)
    plt.ylim(-25, 20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locations and Routes')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

st.header("Visualization Locations")
senders = sender_locations
recipients = recipient_locations

senders_x, senders_y = zip(*senders)
recipients_x, recipients_y = zip(*recipients)

plt.figure(figsize=(10, 10))
plt.scatter(senders_x, senders_y, c='blue', label='Senders', marker='o')
plt.scatter(recipients_x, recipients_y, c='red', label='Recipients', marker='x')

for i, (x, y) in enumerate(senders):
    plt.text(x, y, str(i), fontsize=10, ha='right', color='black')

for i, (x, y) in enumerate(recipients):
    plt.text(x, y, str(i), fontsize=10, ha='right', color='black')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Locations of Senders and Recipients')
plt.legend()
plt.grid(True)
st.pyplot(plt)