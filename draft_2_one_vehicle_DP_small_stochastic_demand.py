# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:44:57 2025

@author: pengr
"""

import numpy as np
import pandas as pd
import itertools
from functools import lru_cache
import matplotlib.pyplot as plt

# Reduced Configuration
time_horizon = 5
discount_factor = 0.9
vehicle_capacity = 8
vehicle_limit = 2
depots = {
    "P1": (0, 0),
    "P2": (8, 8)
}

initial_customers = {
    "I1": {"coord": (1, 2), "demand": 3},
    "I2": {"coord": (4, 2), "demand": 4},
    "I3": {"coord": (2, 5), "demand": 2},
    "I4": {"coord": (5, 5), "demand": 3}
}

new_customers_pool = {
    "I5": {"coord": (6, 1), "appear_time": 2, "base_demand": 3},
    "I6": {"coord": (7, 7), "appear_time": 3, "base_demand": 2}
}

# Step 1: Simulate demand
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

# Build demand table
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

# Step 2: Define Contribution Function
def contribution_function(depot_coord, subset_customers):
    latency = 0.0
    last_point = depot_coord
    arrival_time = 0
    for cid in subset_customers:
        cust_coord = customers[cid]["coord"]
        travel = np.linalg.norm(np.array(last_point) - np.array(cust_coord))
        arrival_time += travel
        latency += arrival_time
        last_point = cust_coord
    return latency

# Step 3: DP Function
decision_trace = {}

@lru_cache(maxsize=None)
def dp_bellman(t, served_mask, v1_left, v2_left, r1, r2):
    if served_mask == (1 << len(customer_ids)) - 1 or t == time_horizon:
        return 0.0

    best_cost = float("inf")
    best_decision = []

    unserved = [cid for i, cid in enumerate(customer_ids) if not (served_mask & (1 << i))]
    current_demands = customer_demand_over_time[t]

    for depot_name, depot_coord in depots.items():
        if (depot_name == "P1" and v1_left == 0) or (depot_name == "P2" and v2_left == 0):
            continue

        capacity_left = {"P1": r1, "P2": r2}[depot_name]

        for r in range(1, len(unserved)+1):
            for subset in itertools.combinations(unserved, r):
                total_demand = sum(current_demands.get(cid, 0) for cid in subset)
                if total_demand > capacity_left:
                    continue

                latency = contribution_function(depot_coord, subset)

                new_mask = served_mask
                for cid in subset:
                    idx = customer_ids.index(cid)
                    new_mask |= (1 << idx)

                if depot_name == "P1":
                    future_cost = dp_bellman(t+1, new_mask, v1_left-1, v2_left, r1-total_demand, r2)
                else:
                    future_cost = dp_bellman(t+1, new_mask, v1_left, v2_left-1, r1, r2-total_demand)

                total_cost = latency + discount_factor * future_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_decision = [{"depot": depot_name, "route": subset, "latency": latency}]

    decision_trace[(t, served_mask, v1_left, v2_left, r1, r2)] = best_decision
    return best_cost

# Step 4: Execute DP
dp_bellman.cache_clear()
optimal_latency = dp_bellman(0, 0, vehicle_limit, vehicle_limit, vehicle_capacity, vehicle_capacity)

# Step 5: Reconstruct solution and state evolution
served_mask = 0
v1_left, v2_left = vehicle_limit, vehicle_limit
r1, r2 = vehicle_capacity, vehicle_capacity

state_evolution = []
# Fix randomness for reproducibility
np.random.seed(20)
for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, r1, r2)
    step = decision_trace.get(key, [])
    if not step:
        break
    route = step[0]
    route_info = {
        "time": t,
        "depot": route["depot"],
        "served_customers": route["route"],
        "latency": route["latency"],
        "remaining_v1": v1_left,
        "remaining_v2": v2_left,
        "cap_r1": r1,
        "cap_r2": r2
    }
    state_evolution.append(route_info)
    for cid in route["route"]:
        idx = customer_ids.index(cid)
        served_mask |= (1 << idx)
    if route["depot"] == "P1":
        v1_left -= 1
        r1 -= sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])
    else:
        v2_left -= 1
        r2 -= sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])

state_evolution_df = pd.DataFrame(state_evolution)

print(state_evolution_df)

# Recover solution_steps from previously reconstructed route logic
solution_steps = []

served_mask = 0
v1_left, v2_left = vehicle_limit, vehicle_limit
r1, r2 = vehicle_capacity, vehicle_capacity

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, r1, r2)
    step = decision_trace.get(key, [])
    for route in step:
        solution_steps.append({
            "time": t,
            "depot": route["depot"],
            "served_customers": route["route"],
            "latency": route["latency"],
            "served_mask_bin": bin(served_mask),
            "v1_left": v1_left,
            "v2_left": v2_left,
            "r1": r1,
            "r2": r2
        })
        for cid in route["route"]:
            idx = customer_ids.index(cid)
            served_mask |= (1 << idx)
        total_demand = sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])
        if route["depot"] == "P1":
            v1_left -= 1
            r1 -= total_demand
        elif route["depot"] == "P2":
            v2_left -= 1
            r2 -= total_demand

solution_steps_df = pd.DataFrame(solution_steps)

# Build the state representation table
state_records = []

for row in solution_steps_df.itertuples():
    t = row.time
    depot = row.depot
    served_customers = row.served_customers
    latency = row.latency
    served_mask_bin = row.served_mask_bin
    v1_left = row.v1_left
    v2_left = row.v2_left
    r1 = row.r1
    r2 = row.r2

    P_t = "{P1, P2}"  # Fixed depots
    V_t = f"{{P1: {v1_left}, P2: {v2_left}}}"
    D_t = f"{served_customers}"
    R_kt = f"{{P1: {r1}, P2: {r2}}}"

    state_string = f"S_{t} = {{ P_t: {P_t}, V_t: {V_t}, D_t: {D_t}, R_k(t): {R_kt} }}"
    state_records.append({"Time": t, "State Representation": state_string})

# Allow full display of wide columns and rows
pd.set_option('display.max_colwidth', None)  # Prevent truncation of column content
pd.set_option('display.width', 200)          # Increase display width
pd.set_option('display.max_columns', None)   # Show all columns
state_df = pd.DataFrame(state_records)
print(state_df)


# Plot depots
plt.figure(figsize=(8, 8))
for depot, coord in depots.items():
    plt.scatter(*coord, color='blue', s=200, label=f'Depot {depot}' if depot == 'P1' else None)
    plt.text(coord[0] + 0.2, coord[1] + 0.2, depot, fontsize=12, color='blue')

# Plot customer locations
for cid, info in customers.items():
    plt.scatter(*info["coord"], color='green', s=100)
    plt.text(info["coord"][0] + 0.2, info["coord"][1] + 0.2, cid, fontsize=10)

# Draw routes from each depot to its assigned customers
colors = {'P1': 'red', 'P2': 'orange'}
for row in solution_steps_df.itertuples():
    depot = row.depot
    d_coord = depots[depot]
    route = list(row.served_customers)
    points = [d_coord] + [customers[cid]["coord"] for cid in route]
    xs, ys = zip(*points)
    plt.plot(xs, ys, color=colors[depot], linestyle='-', linewidth=2, label=f'Route from {depot}' if row.time == 0 else "")

plt.title("Final Routes from Depots to Customers")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

print(f"Total Optimal Latency: {optimal_latency:.2f}")

