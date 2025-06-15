
import json
import numpy as np
import itertools

# Configuration
time_horizon = 5
discount_factor = 0.9
vehicle_capacity = 8
vehicle_limit = 2

# Load simulation data from file
with open("simulation_history.json", "r") as f:
    data = json.load(f)

simulation_history = data["simulation_history"]
customer_ids = data["customer_ids"]
customers = {cid: tuple(coord) for cid, coord in data["customers"].items()}

# Convert simulation history to demand timeline
customer_demand_over_time = {
    t: snap["customers"] for t, snap in enumerate(simulation_history)
}

# Latency calculation function
def compute_latency(route, depot_coord):
    latency = 0
    dist = 0
    last = depot_coord
    for cid in route:
        dist += np.linalg.norm(np.array(last) - np.array(customers[cid]))
        latency += dist
        last = customers[cid]
    return latency

# All feasible routes under capacity
def valid_routes(unserved, demands, cap):
    routes = []
    for i in range(1, len(unserved)+1):
        for subset in itertools.combinations(unserved, i):
            total_demand = sum(demands[cid] for cid in subset)
            if total_demand <= cap:
                routes.append(list(subset))
    return routes

# Brute-force DP-style enumeration
best_total_latency = float("inf")
best_plan = None

depots = {
    "P1": (0, 0),
    "P2": (8, 8)
}

def full_enumerate(t=0, served=set(), plan=[], v1_left=vehicle_limit, v2_left=vehicle_limit, r1=vehicle_capacity, r2=vehicle_capacity):
    global best_total_latency, best_plan
    if t == time_horizon:
        if len(served) == len(customer_ids):
            total_latency = sum(discount_factor**step["time"] * step["latency"] for step in plan)
            if total_latency < best_total_latency:
                best_total_latency = total_latency
                best_plan = plan.copy()
        return

    demands = customer_demand_over_time[t]
    appeared = list(demands.keys())
    unserved = [cid for cid in appeared if cid not in served]

    p1_routes = [[]] if v1_left == 0 else valid_routes(unserved, demands, r1) + [[]]
    p2_routes = [[]] if v2_left == 0 else valid_routes(unserved, demands, r2) + [[]]

    for r1_list in p1_routes:
        for r2_list in p2_routes:
            if set(r1_list).intersection(r2_list):
                continue  # avoid overlap
            if not r1_list and not r2_list:
                continue  # must serve at least someone

            new_plan = plan[:]
            new_served = served.union(r1_list).union(r2_list)
            new_v1, new_v2 = v1_left, v2_left
            new_r1, new_r2 = r1, r2

            if r1_list:
                lat1 = compute_latency(r1_list, depots["P1"])
                demand1 = sum(demands[cid] for cid in r1_list)
                new_plan.append({"time": t, "depot": "P1", "route": r1_list, "latency": lat1})
                new_v1 -= 1
                new_r1 -= demand1

            if r2_list:
                lat2 = compute_latency(r2_list, depots["P2"])
                demand2 = sum(demands[cid] for cid in r2_list)
                new_plan.append({"time": t, "depot": "P2", "route": r2_list, "latency": lat2})
                new_v2 -= 1
                new_r2 -= demand2

            full_enumerate(t+1, new_served, new_plan, new_v1, new_v2, new_r1, new_r2)

# Run
full_enumerate()

if best_plan is None:
    print("❌ No feasible solution found.")
else:
    print(f"\n🔍 Best total latency from enumeration: {best_total_latency:.2f}")
    for step in best_plan:
        print(f"t={step['time']} | Depot={step['depot']} | Route={step['route']} | Latency={step['latency']:.2f}")
