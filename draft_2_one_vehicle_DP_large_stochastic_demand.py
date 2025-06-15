# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:09:57 2025

@author: pengr
"""

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
customer_ids_by_time = {}
all_customer_ids = sorted(dynamic_customers.keys())
timeline = []

for snap in simulation_history:
    row = {"time": snap["time"]}
    appeared_customers = []
    for cid in all_customer_ids:
        demand = snap["customers"].get(cid, np.nan)
        row[cid] = demand
        if not pd.isna(demand):
            appeared_customers.append(cid)
    timeline.append(row)
    customer_ids_by_time[snap["time"]] = appeared_customers

df_timeline = pd.DataFrame(timeline)
customer_demand_over_time = {
    t: {cid: row[cid] for cid in all_customer_ids if not pd.isna(row[cid])}
    for t, row in df_timeline.iterrows()
}

customers = {cid: {"coord": (initial_customers.get(cid) or new_customers_pool[cid])["coord"]}
             for cid in all_customer_ids}

# Step 3: Define Contribution Function
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

# Step 4: DP Function aligned with Bellman Equation
decision_trace = {}

@lru_cache(maxsize=None)
def dp_bellman(t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3):
    if t == time_horizon:
        return 0.0

    best_cost = float("inf")
    best_decision = []

    current_customers = customer_ids_by_time[t]
    current_demands = customer_demand_over_time[t]
    unserved = [cid for i, cid in enumerate(all_customer_ids)
                if cid in current_customers and not (served_mask & (1 << i))]

    for depot_name, depot_coord in depots.items():
        if (depot_name == "P1" and v1_left == 0) or \
           (depot_name == "P2" and v2_left == 0) or \
           (depot_name == "P3" and v3_left == 0):
            continue

        capacity_left = {"P1": r1, "P2": r2, "P3": r3}[depot_name]

        for r in range(1, len(unserved)+1):
            for subset in itertools.combinations(unserved, r):
                total_demand = sum(current_demands.get(cid, 0) for cid in subset)
                if total_demand > capacity_left:
                    continue

                latency = contribution_function(depot_coord, subset)

                new_mask = served_mask
                for cid in subset:
                    idx = all_customer_ids.index(cid)
                    new_mask |= (1 << idx)

                new_v1, new_v2, new_v3 = v1_left, v2_left, v3_left
                new_r1, new_r2, new_r3 = r1, r2, r3
                if depot_name == "P1":
                    new_v1 -= 1
                    new_r1 -= total_demand
                elif depot_name == "P2":
                    new_v2 -= 1
                    new_r2 -= total_demand
                elif depot_name == "P3":
                    new_v3 -= 1
                    new_r3 -= total_demand

                future_cost = dp_bellman(t + 1, new_mask, new_v1, new_v2, new_v3, new_r1, new_r2, new_r3)
                total_cost = latency + discount_factor * future_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_decision = [{"depot": depot_name, "route": subset, "latency": latency}]

    decision_trace[(t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3)] = best_decision
    return best_cost

# Step 5: Execute DP and reconstruct state sequence
dp_bellman.cache_clear()
optimal_latency = dp_bellman(0, 0, vehicle_limit, vehicle_limit, vehicle_limit,
                             vehicle_capacity, vehicle_capacity, vehicle_capacity)

# Extract state trajectory
served_mask = 0
v1_left, v2_left, v3_left = vehicle_limit, vehicle_limit, vehicle_limit
r1, r2, r3 = vehicle_capacity, vehicle_capacity, vehicle_capacity

state_trace = []

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3)
    step = decision_trace.get(key, [])
    for route in step:
        entry = {
            "time": t,
            "depot": route["depot"],
            "served_customers": route["route"],
            "latency": round(route["latency"], 2),
            "v1_left": v1_left,
            "v2_left": v2_left,
            "v3_left": v3_left,
            "r1": r1,
            "r2": r2,
            "r3": r3
        }
        state_trace.append(entry)
        for cid in route["route"]:
            idx = all_customer_ids.index(cid)
            served_mask |= (1 << idx)
        if route["depot"] == "P1":
            v1_left -= 1
            r1 -= sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])
        elif route["depot"] == "P2":
            v2_left -= 1
            r2 -= sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])
        elif route["depot"] == "P3":
            v3_left -= 1
            r3 -= sum(customer_demand_over_time[t].get(cid, 0) for cid in route["route"])

df_state_trace = pd.DataFrame(state_trace)
print(df_state_trace)
