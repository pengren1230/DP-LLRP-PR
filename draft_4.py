# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:08:07 2025

@author: pengr
"""
import heapq
import itertools
from math import sqrt
from itertools import count
import pandas as pd
import time

# Problem Parameters
vehicle_capacity = 70
total_vehicle_limit = 5  # Global limit

depot_coords = [(6, 7), (19, 44), (37, 23), (35, 6), (5, 8)]
customer_coords = [
    (20, 35), (8, 31), (29, 43), (18, 39), (19, 47), (31, 24), (38, 50), (33, 21), (2, 27), (1, 12),
    (26, 20), (20, 33), (15, 46), (20, 26), (17, 19), (15, 12), (5, 30), (13, 40), (38, 5), (9, 40)
]
customer_demands = [
    17, 18, 13, 19, 12, 18, 13, 13, 17, 20,
    16, 18, 15, 11, 18, 16, 15, 15, 15, 16
]
num_depots = len(depot_coords)
num_customers = len(customer_coords)
depot_ids = [f"D{i+1}" for i in range(num_depots)]
customer_ids = [f"C{i+1}" for i in range(num_customers)]

# Create distance matrix
all_coords = depot_coords + customer_coords
distance_matrix = [
    [round(sqrt((x1 - x2)**2 + (y1 - y2)**2), 2) for x2, y2 in all_coords]
    for x1, y1 in all_coords
]

# Build data dicts
depots = {
    did: {"coord": depot_coords[i], "index": i}
    for i, did in enumerate(depot_ids)
}
customers = {
    cid: {"coord": customer_coords[i], "demand": customer_demands[i], "index": i + num_depots}
    for i, cid in enumerate(customer_ids)
}

# Bitmask helpers
def is_served(mask, idx): return (mask >> idx) & 1
def set_served(mask, idx): return mask | (1 << idx)

# Initial state
initial_state = {
    "served_mask": 0,
    "vehicle_used": 0,
    "latency": 0,
    "routes": [],
}

# Greedy route expansion
def greedy_next_step(state):
    served_mask = state["served_mask"]
    vehicles_used = state["vehicle_used"]
    next_states = []

    if vehicles_used >= total_vehicle_limit:
        return []

    for depot_idx, depot_id in enumerate(depot_ids):
        unserved = [customer_ids[i] for i in range(num_customers) if not is_served(served_mask, i)]

# try highter number other than 4, may give a better solution
        for r in range(1, min(10, len(unserved)) + 1):
            for route in itertools.combinations(unserved, r):
                total_demand = sum(customers[c]["demand"] for c in route)
                if total_demand > vehicle_capacity:
                    continue

                depot_global_idx = depots[depot_id]["index"]
                curr = depot_global_idx
                acc_time = 0
                route_latency = 0
                for c in route:
                    next_idx = customers[c]["index"]
                    dist = distance_matrix[curr][next_idx]
                    acc_time += dist
                    route_latency += acc_time
                    curr = next_idx

                new_mask = served_mask
                for c in route:
                    new_mask = set_served(new_mask, customer_ids.index(c))

                new_routes = state["routes"] + [{
                    "depot": depot_id,
                    "served_customers": route,
                    "latency": route_latency,
                    "route_coordinates": [depots[depot_id]["coord"]] + [customers[c]["coord"] for c in route]
                }]

                next_states.append({
                    "served_mask": new_mask,
                    "vehicle_used": vehicles_used + 1,
                    "latency": state["latency"] + route_latency,
                    "routes": new_routes
                })

    return next_states

# Rollout DP
def rollout_dp(max_depth=3, timeout_seconds=600):
    pq = []
    tie_counter = count()
    heapq.heappush(pq, (0, next(tie_counter), initial_state))
    MAX_QUEUE_SIZE = 500

    best_final_state = None
    min_latency = float("inf")
    
    
    start_time = time.time()
    


    while pq:
        if time.time() - start_time > timeout_seconds:
            print("⏰ Timeout reached. Returning best solution found so far.")
            return best_final_state

        _, _, state = heapq.heappop(pq)

        if state["served_mask"] == (1 << num_customers) - 1:
            if state["latency"] < min_latency:
                min_latency = state["latency"]
                best_final_state = state
            continue

        if len(state["routes"]) >= max_depth * total_vehicle_limit:
            continue

        for next_state in greedy_next_step(state):
            heapq.heappush(pq, (next_state["latency"], next(tie_counter), next_state))

        if len(pq) > MAX_QUEUE_SIZE:
            pq = heapq.nsmallest(MAX_QUEUE_SIZE, pq)

    elapsed_time = round(time.time() - start_time, 2)
    return best_final_state, elapsed_time



# Execute with global vehicle limit
best_result, elapsed_time = rollout_dp(max_depth=3, timeout_seconds=600)
print(f"⏱ Finished in {elapsed_time} seconds")


if best_result is not None:
    routes_df = pd.DataFrame(best_result["routes"])
    routes_df["Vehicle_ID"] = routes_df.index + 1
    import ace_tools as tools; tools.display_dataframe_to_user(name="Limited 5 Vehicle Routes", dataframe=routes_df)
else:
    print("❌ No solution found.")
