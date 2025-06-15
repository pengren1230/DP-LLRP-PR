# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:23:41 2025

@author: pengr
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import time


def parse_llrp_data(file_path):
    def parse_numbers(line):
        return list(map(float, line.strip().split()))

    def collect_values(lines, start_idx, count):
        values = []
        idx = start_idx
        while len(values) < count and idx < len(lines):
            line = lines[idx].strip()
            if line:
                values.append(float(line))
            idx += 1
        return values, idx

    with open(file_path, 'r') as f:
        lines = f.readlines()

    line_idx = 0
    num_customers = int(lines[line_idx].strip()); line_idx += 1
    num_depots = int(lines[line_idx].strip()); line_idx += 1
    while lines[line_idx].strip() == '':
        line_idx += 1

    depots_coord = []
    while len(depots_coord) < num_depots:
        depots_coord.append(parse_numbers(lines[line_idx].strip()))
        line_idx += 1

    customers_coord = []
    while len(customers_coord) < num_customers:
        customers_coord.append(parse_numbers(lines[line_idx].strip()))
        line_idx += 1

    while lines[line_idx].strip() == '':
        line_idx += 1
    vehicle_capacity = int(lines[line_idx].strip()); line_idx += 1

    depot_capacities, line_idx = collect_values(lines, line_idx, num_depots)
    customer_demands, line_idx = collect_values(lines, line_idx, num_customers)
    depot_opening_costs, line_idx = collect_values(lines, line_idx, num_depots)
    while lines[line_idx].strip() == '':
        line_idx += 1
    route_opening_cost = float(lines[line_idx].strip()); line_idx += 1
    while line_idx < len(lines) and lines[line_idx].strip() == '':
        line_idx += 1
    cost_type = int(lines[line_idx].strip()); line_idx += 1

    return {
        "depots_coord": depots_coord,
        "customers_coord": customers_coord,
        "customer_demands": customer_demands,
        "vehicle_capacity": vehicle_capacity
    }

def compute_distance_matrix(depots, customers):
    points = depots + customers
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
    return dist

def total_latency(route, depot_index, dist, demand=None):
    time = 0
    latency = 0
    current = depot_index
    for cust in route:
        time += dist[current][cust]
        latency += time if demand is None else time * demand[cust - len(depot_coord)]
        current = cust
    return latency

def solve_llrp_brute(instance, timeout_seconds=1800):
    global depot_coord
    depot_coord = instance["depots_coord"]
    customer_coord = instance["customers_coord"]
    customer_demand = instance["customer_demands"]
    vehicle_cap = instance["vehicle_capacity"]

    N_d, N_c = len(depot_coord), len(customer_coord)
    dist = compute_distance_matrix(depot_coord, customer_coord)
    best_total_latency = float('inf')
    best_solution = None

    start_time = time.time()  # Start timer

    for open_depots in itertools.product([0, 1], repeat=N_d):
        # Check for timeout
        if time.time() - start_time > timeout_seconds:
            print("⏰ Timeout reached. Returning best solution found so far.")
            return best_total_latency, best_solution

        if sum(open_depots) == 0:
            continue

        open_depot_ids = [i for i, flag in enumerate(open_depots) if flag]
        depot_routes = {d: [] for d in open_depot_ids}

        for cust in range(len(customer_coord)):
            best_d = min(open_depot_ids, key=lambda d: dist[d][cust])
            depot_routes[best_d].append(cust)

        total_lat = 0

        for depot_id, cust_list in depot_routes.items():
            min_lat = float('inf')
            for perm in itertools.permutations(cust_list):
                if time.time() - start_time > timeout_seconds:
                    print("⏰ Timeout reached during routing permutations.")
                    return best_total_latency, best_solution

                lat = total_latency(perm, depot_id, dist, customer_demand)
                if lat < min_lat:
                    min_lat = lat
            total_lat += min_lat


        if total_lat < best_total_latency:
            best_total_latency = total_lat
            best_solution = (open_depots, depot_routes)

    return best_total_latency, best_solution

data = parse_llrp_data(r"C:\Users\pengr\OneDrive\Desktop\DP_material\TB_data\coordP111112.dat")
latency, solution = solve_llrp_brute(data, timeout_seconds=3600)
print("Best latency found:", latency)


print("Best Total Latency:", latency)
print("Depot Configuration:", solution[0])
print("Routes per Depot:", solution[1])
