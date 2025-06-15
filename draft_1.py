# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:22:12 2025

@author: pengr
"""

# Re-import required modules and restore state after kernel reset
import numpy as np
from itertools import permutations, combinations

# Redefine depots and customers
depots = {
    "P1": (0, 0),
    "P3": (5, 5)
}

customers = {
    "I1": {"coord": (2, 3), "demand": 5},
    "I2": {"coord": (6, 7), "demand": 7},
    "I3": {"coord": (9, 4), "demand": 4},
    "I4": {"coord": (3, 8), "demand": 6},
    "I5": {"coord": (7, 2), "demand": 8},
}

vehicle_capacity = 10

# Build distance matrix
all_points = {**{k: depots[k] for k in depots}, **{k: v["coord"] for k, v in customers.items()}}
point_names = list(all_points.keys())
coord_array = np.array([all_points[name] for name in point_names])
dist_matrix = np.linalg.norm(coord_array[:, None] - coord_array[None, :], axis=2)
point_index = {name: idx for idx, name in enumerate(point_names)}

# Latency calculation from one depot
def calculate_latency(route, start_depot):
    idx_depot = point_index[start_depot]
    latency = 0
    arrival_time = 0
    load = 0
    prev_idx = idx_depot

    for cust in route:
        cust_idx = point_index[cust]
        travel = dist_matrix[prev_idx, cust_idx]
        arrival_time += travel
        latency += arrival_time
        load += customers[cust]["demand"]
        if load > vehicle_capacity:
            return float("inf")
        prev_idx = cust_idx

    return latency

def calculate_total_latency_multi(routes, depot):
    total_latency = 0
    for route in routes:
        latency = calculate_latency(route, depot)
        if latency == float("inf"):
            return float("inf")
        total_latency += latency
    return total_latency

# Generate all 2-group splits
def all_partitions(customers, max_parts=2):
    if max_parts == 1:
        yield [tuple(customers)]
        return
    for i in range(1, len(customers)):
        for group in combinations(customers, i):
            remaining = list(set(customers) - set(group))
            for parts in all_partitions(remaining, max_parts - 1):
                yield [group] + parts

# Main solver allowing 2 vehicles per depot
def solve_llrp_multi_vehicle():
    cust_names = list(customers.keys())
    best_latency = float("inf")
    best_config = None

    for r in range(1, len(cust_names)):
        for comb in combinations(cust_names, r):
            group1 = list(comb)
            group2 = list(set(cust_names) - set(group1))

            for part1 in all_partitions(group1, max_parts=2):
                routes1 = [list(p) for p in part1]

                for part2 in all_partitions(group2, max_parts=2):
                    routes2 = [list(p) for p in part2]

                    lat1 = calculate_total_latency_multi(routes1, "P1")
                    lat2 = calculate_total_latency_multi(routes2, "P3")
                    total = lat1 + lat2
                    if total < best_latency:
                        best_latency = total
                        best_config = {"P1": routes1, "P3": routes2}

    return best_latency, best_config

latency_multi, assignment_multi = solve_llrp_multi_vehicle()
latency_multi, assignment_multi
