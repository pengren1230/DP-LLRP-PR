# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 09:42:27 2025

@author: pengr
"""

# Rewriting the existing Python implementation with a formal structure
# Includes pseudocode at the top and aligns code with DP formulation

pseudocode = """
Pseudocode for Bellman-style DP to Solve LLRP with Vehicle Capacity and Uncertainty

State:
    S_t = {P_t, V_t, D_t, R_k(t)} where:
        - P_t: depot status (fixed in this case)
        - V_t: vehicle count available per depot
        - D_t: demand at customer locations
        - R_k(t): remaining vehicle capacity for each depot's vehicle

Action:
    a_t = {depot, [customers], g} where:
        - depot: selected depot
        - [customers]: subset of customers to serve
        - g: total demand assigned, must be ≤ capacity

Contribution Function:
    C(S_t, a_t) = sum of cumulative arrival times for each customer in route from depot

Bellman Recursion:
    V(S_t) = min_a_t [ C(S_t, a_t) + β * E[V(S_{t+1})] ]
"""

print(pseudocode)

# Then we output the full rewritten code
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
@lru_cache(maxsize=None)
def dp_bellman(t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3):
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

        capacity_left = {"P1": r1, "P2": r2, "P3": r3}[depot_name]

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

# Step 5: Execute DP and Collect Routes
decision_trace = {}
dp_bellman.cache_clear()
optimal_latency = dp_bellman(
    0, 0, vehicle_limit, vehicle_limit, vehicle_limit,
    vehicle_capacity, vehicle_capacity, vehicle_capacity
)

print("\n✅ Optimal Latency:", optimal_latency)

# Step 6: Reconstruct the DP solution path
served_mask = 0
v1_left, v2_left, v3_left = vehicle_limit, vehicle_limit, vehicle_limit
r1, r2, r3 = vehicle_capacity, vehicle_capacity, vehicle_capacity

solution_trace = []

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3)
    step = decision_trace.get(key, [])
    if not step:
        break

    for route in step:
        depot = route["depot"]
        served_customers = route["route"]
        latency = route["latency"]

        solution_trace.append({
            "Time": t,
            "Depot": depot,
            "Served Customers": served_customers,
            "Latency": round(latency, 2),
            "Depot Remaining Vehicles": {
                "P1": v1_left, "P2": v2_left, "P3": v3_left
            }[depot],
            "Depot Remaining Capacity": {
                "P1": r1, "P2": r2, "P3": r3
            }[depot]
        })

        # Update state
        for cid in served_customers:
            idx = customer_ids.index(cid)
            served_mask |= (1 << idx)

        total_demand = sum(customer_demand_over_time[t].get(cid, 0) for cid in served_customers)
        if depot == "P1":
            v1_left -= 1
            r1 -= total_demand
        elif depot == "P2":
            v2_left -= 1
            r2 -= total_demand
        elif depot == "P3":
            v3_left -= 1
            r3 -= total_demand

# Output full trace
df_trace = pd.DataFrame(solution_trace)
print("\n🔎 DP Route Decisions:")
print(df_trace.to_string(index=False))


# Step 7: Print state evolution over time
state_trace = []

served_mask = 0
v1_left, v2_left, v3_left = vehicle_limit, vehicle_limit, vehicle_limit
r1, r2, r3 = vehicle_capacity, vehicle_capacity, vehicle_capacity

for t in range(time_horizon):
    key = (t, served_mask, v1_left, v2_left, v3_left, r1, r2, r3)
    step = decision_trace.get(key, [])
    if not step:
        break

    served_customers = step[0]["route"] if step else []
    state_trace.append({
        "Time": t,
        "Served Mask": bin(served_mask),
        "Served (Readable)": [cid for i, cid in enumerate(customer_ids) if (served_mask & (1 << i))],
        "Depot Vehicle Count": {"P1": v1_left, "P2": v2_left, "P3": v3_left},
        "Depot Remaining Capacity": {"P1": r1, "P2": r2, "P3": r3},
        "Just Served Customers": served_customers
    })

    total_demand = sum(customer_demand_over_time[t].get(cid, 0) for cid in served_customers)
    for cid in served_customers:
        idx = customer_ids.index(cid)
        served_mask |= (1 << idx)

    depot = step[0]["depot"]
    if depot == "P1":
        v1_left -= 1
        r1 -= total_demand
    elif depot == "P2":
        v2_left -= 1
        r2 -= total_demand
    elif depot == "P3":
        v3_left -= 1
        r3 -= total_demand

# Output DP state history
df_state = pd.DataFrame(state_trace)
print("\n🧠 DP States Over Time:")
print(df_state.to_string(index=False))

