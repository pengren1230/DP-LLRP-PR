# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:33:54 2025

@author: pengr
"""

# Consolidated full code including simulation, DP with vehicle limits, and result display

import numpy as np
import pandas as pd
import itertools
from functools import lru_cache

# Configuration
time_horizon = 10
discount_factor = 0.9
vehicle_capacity = 15
vehicle_limit = 1
depots = {
    "P1": (0, 0),
    "P2": (10, 10),
    "P3": (5, 5)
}

initial_customers = {
    "I1": {"coord": (2, 3), "demand": 5},
    "I2": {"coord": (6, 7), "demand": 7},
    "I3": {"coord": (9, 4), "demand": 4},
    "I4": {"coord": (3, 8), "demand": 6},
    "I5": {"coord": (7, 2), "demand": 8}
}

new_customers_pool = {
    "I6": {"coord": (4, 6), "appear_time": 1, "base_demand": 3},
    "I7": {"coord": (8, 6), "appear_time": 3, "base_demand": 5},
    "I8": {"coord": (1, 9), "appear_time": 5, "base_demand": 4},
    "I9": {"coord": (6, 1), "appear_time": 6, "base_demand": 6},
    "I10": {"coord": (9, 9), "appear_time": 8, "base_demand": 2}
}

# Step 1: Simulate demand (monotonically increasing)
def update_demand(d):
    return d + np.random.choice([0, 1])

dynamic_customers = initial_customers.copy()
simulation_history = []

for t in range(time_horizon):
    for cid, data in new_customers_pool.items():
        if data["appear_time"] == t:
            dynamic_customers[cid] = {"coord": data["coord"], "demand": data["base_demand"]}
    for cid in dynamic_customers:
        dynamic_customers[cid]["demand"] = update_demand(dynamic_customers[cid]["demand"])
    snapshot = {
        "time": t,
        "customers": {cid: dynamic_customers[cid]["demand"] for cid in dynamic_customers}
    }
    simulation_history.append(snapshot)

# Step 2: Build demand table
customer_ids = sorted(dynamic_customers.keys())
timeline = []
for snap in simulation_history:
    row = {"time": snap["time"]}
    for cid in customer_ids:
        row[cid] = snap["customers"].get(cid, np.nan)
    timeline.append(row)
df_timeline = pd.DataFrame(timeline)

customer_demand_over_time = {
    t: {cid: row[cid] for cid in customer_ids if not pd.isna(row[cid])}
    for t, row in df_timeline.iterrows()
}

customers = {cid: {"coord": (initial_customers.get(cid) or new_customers_pool[cid])["coord"]}
             for cid in customer_ids}

# Step 3: Define DP with vehicle usage constraints
@lru_cache(maxsize=None)
def dp_with_vehicle_limit(t, served_mask, v1_left, v2_left, v3_left):
    if served_mask == (1 << len(customer_ids)) - 1 or t == time_horizon:
        return 0.0

    best_cost = float("inf")
    best_decision = []

    unserved = [cid for i, cid in enumerate(customer_ids) if not (served_mask & (1 << i))]
    current_demands = customer_demand_over_time[t]

    for depot_name, depot_coord in depots.items():
        if (depot_name == "P1" and v1_left == 0) or \
           (depot_name == "P2" and v2_left == 0) or \
           (depot_name == "P3" and v3_left == 0):
            continue

        for r in range(1, len(unserved)+1):
            for subset in itertools.combinations(unserved, r):
                if sum(current_demands.get(cid, 0) for cid in subset) > vehicle_capacity:
                    continue

                latency = 0.0
                last_point = depot_coord
                arrival_time = 0
                for cid in subset:
                    cust_coord = customers[cid]["coord"]
                    travel = np.linalg.norm(np.array(last_point) - np.array(cust_coord))
                    arrival_time += travel
                    latency += arrival_time
                    last_point = cust_coord

                new_mask = served_mask
                for cid in subset:
                    idx = customer_ids.index(cid)
                    new_mask |= (1 << idx)

                new_v1 = v1_left - 1 if depot_name == "P1" else v1_left
                new_v2 = v2_left - 1 if depot_name == "P2" else v2_left
                new_v3 = v3_left - 1 if depot_name == "P3" else v3_left

                future_cost = dp_with_vehicle_limit(t + 1, new_mask, new_v1, new_v2, new_v3)
                total_cost = latency + discount_factor * future_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_decision = [{"depot": depot_name, "route": subset, "latency": latency}]

    decision_trace_with_limit[(t, served_mask, v1_left, v2_left, v3_left)] = best_decision
    return best_cost

# Step 4: Execute DP and extract final routes
decision_trace_with_limit = {}
dp_with_vehicle_limit.cache_clear()
optimal_latency_limited = dp_with_vehicle_limit(0, 0, vehicle_limit, vehicle_limit, vehicle_limit)

served_mask = 0
v1_left, v2_left, v3_left = vehicle_limit, vehicle_limit, vehicle_limit
final_routes_limited = []

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, v3_left)
    step = decision_trace_with_limit.get(key, [])
    for route in step:
        final_routes_limited.append({
            "time": t,
            "depot": route["depot"],
            "served_customers": route["route"],
            "latency": route["latency"]
        })
        for cid in route["route"]:
            idx = customer_ids.index(cid)
            served_mask |= (1 << idx)
        if route["depot"] == "P1":
            v1_left -= 1
        elif route["depot"] == "P2":
            v2_left -= 1
        elif route["depot"] == "P3":
            v3_left -= 1

final_routes_limited_df = pd.DataFrame(final_routes_limited)
# Print or export the result in your local Python environment
print(final_routes_limited_df)
# final_routes_limited_df.to_csv("final_routes_output.csv", index=False)

print("Optimal Latency:", optimal_latency_limited)


optimal_latency_limited

# Reconstruct route details including full path (depot -> customer sequence)
detailed_routes = []

served_mask = 0
v1_left, v2_left, v3_left = vehicle_limit, vehicle_limit, vehicle_limit

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, v3_left)
    step = decision_trace_with_limit.get(key, [])
    for route in step:
        route_path = [depots[route["depot"]]]  # Start from depot coordinates
        for cid in route["route"]:
            route_path.append(customers[cid]["coord"])
        detailed_routes.append({
            "time": t,
            "depot": route["depot"],
            "served_customers": route["route"],
            "latency": route["latency"],
            "route_coordinates": route_path
        })
        for cid in route["route"]:
            idx = customer_ids.index(cid)
            served_mask |= (1 << idx)
        if route["depot"] == "P1":
            v1_left -= 1
        elif route["depot"] == "P2":
            v2_left -= 1
        elif route["depot"] == "P3":
            v3_left -= 1

# Convert to DataFrame for display
detailed_routes_df = pd.DataFrame(detailed_routes)
print(detailed_routes_df)
# Or to save as CSV:
# detailed_routes_df.to_csv("detailed_routes_output.csv", index=False)

import matplotlib.pyplot as plt

# Assign a unique color per depot
depot_colors = {"P1": "red", "P2": "blue", "P3": "green"}

# Plot each route over time
plt.figure(figsize=(10, 8))
for entry in detailed_routes:
    coords = entry["route_coordinates"]
    x_vals, y_vals = zip(*coords)
    depot = entry["depot"]
    plt.plot(x_vals, y_vals, marker='o', label=f"{depot} - t{entry['time']}", color=depot_colors[depot])

    # Label the start point (depot) and customers
    plt.text(x_vals[0], y_vals[0], f"{depot}", fontsize=9, ha='right', va='bottom')
    for i in range(1, len(x_vals)):
        customer_label = entry["served_customers"][i-1]
        plt.text(x_vals[i], y_vals[i], customer_label, fontsize=8, ha='left', va='top')

# Plot all depots
for d, coord in depots.items():
    plt.scatter(*coord, c=depot_colors[d], s=100, edgecolor='black', label=f"{d} (Depot)")

plt.title("Routes from Depots to Customers over 10 Periods")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
