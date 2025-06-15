# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 13:49:48 2025

@author: pengr
"""

import numpy as np
import pandas as pd
import itertools


# Problem setup
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


np.random.seed(20)

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

customer_ids = sorted(dynamic_customers.keys())
timeline = []
for snap in simulation_history:
    row = {"time": snap["time"]}
    for cid in customer_ids:
        row[cid] = snap["customers"].get(cid, np.nan)
    timeline.append(row)

df_timeline = pd.DataFrame(timeline)

customers = {cid: {"coord": (initial_customers.get(cid) or new_customers_pool[cid])["coord"]}
             for cid in customer_ids}

customer_demand_over_time = {
    t: {cid: row[cid] for cid in customer_ids if not pd.isna(row[cid])}
    for t, row in df_timeline.iterrows()
}



customer_demand_over_time = {
    t: {cid: row[cid] for cid in customer_ids if not pd.isna(row[cid])}
    for t, row in df_timeline.iterrows()
}


# Latency calculation function
def compute_latency(route, depot_coord):
    latency = 0
    dist = 0
    last = depot_coord
    for cid in route:
        dist += np.linalg.norm(np.array(last) - np.array(customers[cid]["coord"]))
        latency += dist
        last = customers[cid]["coord"]
    return latency

# All subsets of customers (excluding empty set)
def all_subsets(customers_list):
    return [list(x) for i in range(1, len(customers_list)+1) for x in itertools.combinations(customers_list, i)]

# Brute-force solver
best_total_latency = float("inf")
best_plan = None

def enumerate_plans(t=0, served=set(), plan=[], v1_left=vehicle_limit, v2_left=vehicle_limit, r1=vehicle_capacity, r2=vehicle_capacity):
    global best_total_latency, best_plan
    if t == time_horizon:
        if len(served) == len(customer_ids):  # ✅ Only accept complete solutions
            total_latency = sum(discount_factor**step["time"] * step["latency"] for step in plan)
            if total_latency < best_total_latency:
                best_total_latency = total_latency
                best_plan = plan.copy()
        return


    demands = customer_demand_over_time[t]

    # ✅ Only consider customers who have appeared
    available_customers = list(demands.keys())
    unserved = [cid for cid in available_customers if cid not in served]
    
    any_feasible_route = False

    for depot in ["P1", "P2"]:
        if (depot == "P1" and v1_left == 0) or (depot == "P2" and v2_left == 0):
            continue
        cap_left = r1 if depot == "P1" else r2
        depot_coord = depots[depot]

        for subset in all_subsets(unserved):
            demand_sum = sum(demands.get(cid, 0) for cid in subset)
            if demand_sum > cap_left:
                continue
            any_feasible_route = True
            latency = compute_latency(subset, depot_coord)
            new_plan = plan + [{
                "time": t,
                "depot": depot,
                "route": subset,
                "latency": latency
            }]
            new_served = served.union(subset)
            if depot == "P1":
                enumerate_plans(t+1, new_served, new_plan, v1_left-1, v2_left, r1-demand_sum, r2)
            else:
                enumerate_plans(t+1, new_served, new_plan, v1_left, v2_left-1, r1, r2-demand_sum)

    if not any_feasible_route:
        enumerate_plans(t+1, served, plan, v1_left, v2_left, r1, r2)



# Start the brute-force enumeration
enumerate_plans()

# Print results
print(f"\n🔍 Best total latency from enumeration: {best_total_latency:.2f}")
for step in best_plan:
    print(f"t={step['time']} | Depot={step['depot']} | Route={step['route']} | Latency={step['latency']:.2f}")
