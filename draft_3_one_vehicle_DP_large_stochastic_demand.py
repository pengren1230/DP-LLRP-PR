# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 18:07:54 2025

@author: pengr
"""

"""DP approach for large LLRP instance.
Based on logic from `draft_2_one_vehicle_DP_small_stochastic_demand.py` but
adapted for a static problem with multiple depots and more customers.
"""

from functools import lru_cache
import itertools
from math import sqrt
import time

# Problem Parameters
vehicle_capacity = 70
vehicles_per_depot = 5

# Limit exploration
MAX_ROUTE_SIZE = 3  # restrict route size to keep DP manageable
# Allow DP search up to 30 minutes before falling back
TIME_LIMIT = 30 * 120  # seconds
MAX_COMBINATIONS = 100  # limit subsets evaluated per route size
start_time = time.time()

# Coordinates
depot_coords = [(6, 7), (19, 44), (37, 23), (35, 6), (5, 8)]
customer_coords = [
    (20, 35), (8, 31), (29, 43), (18, 39), (19, 47), (31, 24), (38, 50),
    (33, 21), (2, 27), (1, 12), (26, 20), (20, 33), (15, 46), (20, 26),
    (17, 19), (15, 12), (5, 30), (13, 40), (38, 5), (9, 40)
]
customer_demands = [
    17, 18, 13, 19, 12, 18, 13, 13, 17, 20,
    16, 18, 15, 11, 18, 16, 15, 15, 15, 16
]

num_depots = len(depot_coords)
num_customers = len(customer_coords)

depot_ids = [f"D{i+1}" for i in range(num_depots)]
customer_ids = [f"C{i+1}" for i in range(num_customers)]

# Distance matrix for latency calculation
all_coords = depot_coords + customer_coords

distance_matrix = [
    [round(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 2) for x2, y2 in all_coords]
    for x1, y1 in all_coords
]

depots = {
    did: {"coord": depot_coords[i], "index": i}
    for i, did in enumerate(depot_ids)
}
customers = {
    cid: {"coord": customer_coords[i], "demand": customer_demands[i], "index": i + num_depots}
    for i, cid in enumerate(customer_ids)
}

# Helper for latency from a depot to a subset of customers
def contribution_function(depot_coord, subset_customers):
    latency = 0.0
    last_point = depot_coord
    arrival = 0.0
    for cid in subset_customers:
        c_coord = customers[cid]["coord"]
        dist = sqrt((last_point[0] - c_coord[0]) ** 2 + (last_point[1] - c_coord[1]) ** 2)
        arrival += dist
        latency += arrival
        last_point = c_coord
    return latency

# Decision trace for policy reconstruction
decision_trace = {}

class TimeLimitExceeded(Exception):
    """Raised when DP exceeds the pre-defined time limit."""
    pass

discount_factor = 0.9
# Maximum number of routes possible (upper bound)
time_horizon = vehicles_per_depot * num_depots

@lru_cache(maxsize=None)
def dp_bellman(t, served_mask, vehicles_state, capacity_state):
    # Stop if time limit exceeded
    if time.time() - start_time > TIME_LIMIT:
        raise TimeLimitExceeded
    if served_mask == (1 << num_customers) - 1 or t == time_horizon:
        return 0.0

    vehicles = list(vehicles_state)
    capacities = list(capacity_state)

    # Reset capacities if previous vehicle finished and more are available
    for i in range(num_depots):
        if capacities[i] == 0 and vehicles[i] > 0:
            capacities[i] = vehicle_capacity

    unserved = [customer_ids[i] for i in range(num_customers) if not (served_mask & (1 << i))]

    best_cost = float("inf")
    best_decision = []

    for d_index, d_id in enumerate(depot_ids):
        if vehicles[d_index] == 0:
            continue
        capacity_left = capacities[d_index]
        for r in range(1, min(len(unserved), MAX_ROUTE_SIZE) + 1):
            combo_count = 0
            for subset in itertools.combinations(unserved, r):
                if time.time() - start_time > TIME_LIMIT:
                    raise TimeLimitExceeded
                if combo_count >= MAX_COMBINATIONS:
                    break
                combo_count += 1
                total_demand = sum(customers[c]["demand"] for c in subset)
                if total_demand > capacity_left:
                    continue
                latency = contribution_function(depots[d_id]["coord"], subset)

                new_mask = served_mask
                for cid in subset:
                    idx = customer_ids.index(cid)
                    new_mask |= (1 << idx)

                new_vehicles = vehicles.copy()
                new_capacities = capacities.copy()
                new_vehicles[d_index] -= 1
                new_capacities[d_index] -= total_demand
                if new_capacities[d_index] == 0 and new_vehicles[d_index] > 0:
                    new_capacities[d_index] = vehicle_capacity

                future = dp_bellman(t + 1, new_mask, tuple(new_vehicles), tuple(new_capacities))
                total_cost = latency + discount_factor * future
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_decision = [{"depot": d_id, "route": subset, "latency": latency}]

    decision_trace[(t, served_mask, vehicles_state, capacity_state)] = best_decision
    return best_cost

# Execute DP
initial_vehicles = tuple([vehicles_per_depot] * num_depots)
initial_cap = tuple([vehicle_capacity] * num_depots)

dp_bellman.cache_clear()
dp_success = True
try:
    optimal_latency = dp_bellman(0, 0, initial_vehicles, initial_cap)
except TimeLimitExceeded:
    optimal_latency = float('nan')
    dp_success = False
    print("Time limit exceeded during DP search. Returning partial solution.")

    def greedy_solution():
        served = set()
        steps = []
        vehicles = [vehicles_per_depot] * num_depots
        capacities = [vehicle_capacity] * num_depots
        t = 0
        for d_idx, d_id in enumerate(depot_ids):
            while vehicles[d_idx] > 0:
                route = []
                cap_left = capacities[d_idx]
                for cid in customer_ids:
                    if cid in served:
                        continue
                    demand = customers[cid]['demand']
                    if demand <= cap_left:
                        route.append(cid)
                        cap_left -= demand
                        served.add(cid)
                if not route:
                    break
                latency = contribution_function(depots[d_id]['coord'], route)
                steps.append({'time': t, 'depot': d_id, 'served_customers': tuple(route),
                              'latency': latency})
                vehicles[d_idx] -= 1
                t += 1
        return steps

    solution_steps = greedy_solution()

if dp_success:
    served_mask = 0
    vehicles_state = list(initial_vehicles)
    capacity_state = list(initial_cap)

    solution_steps = []
    for t in range(time_horizon):
        key = (t, served_mask, tuple(vehicles_state), tuple(capacity_state))
        step = decision_trace.get(key)
        if not step:
            break
        route = step[0]
        solution_steps.append({
            "time": t,
            "depot": route["depot"],
            "served_customers": route["route"],
            "latency": route["latency"],
            "vehicles_left": vehicles_state.copy(),
            "capacities": capacity_state.copy()
        })
        for cid in route["route"]:
            idx = customer_ids.index(cid)
            served_mask |= (1 << idx)
        d_idx = depot_ids.index(route["depot"])
        total_demand = sum(customers[c]["demand"] for c in route["route"])
        vehicles_state[d_idx] -= 1
        capacity_state[d_idx] -= total_demand
        if capacity_state[d_idx] == 0 and vehicles_state[d_idx] > 0:
            capacity_state[d_idx] = vehicle_capacity

for step in solution_steps:
    print(step)
if optimal_latency == optimal_latency:  # check for NaN
    print(f"Optimal Latency: {optimal_latency:.2f}")
total_latency = sum(step['latency'] for step in solution_steps)
print(f"Total Latency: {total_latency:.2f}")