# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 12:08:41 2025

@author: pengr
"""

# Define file path to your .dat file
file_path = "C:\\Users\\pengr\\OneDrive\\Desktop\\DP_material\\TB_data\\coordP111112.dat"

# Read the file lines
with open(file_path, 'r') as f:
    lines = f.readlines()

# Helper function to collect non-empty float values
def collect_values(lines, start_idx, count):
    values = []
    idx = start_idx
    while len(values) < count and idx < len(lines):
        line = lines[idx].strip()
        if line != '':
            values.append(float(line))
        idx += 1
    return values, idx

# Helper to parse tab-separated coordinates
def parse_numbers(line):
    return list(map(float, line.strip().split()))

# Start parsing
line_idx = 0
num_customers = int(lines[line_idx].strip()); line_idx += 1
num_depots = int(lines[line_idx].strip()); line_idx += 1

# Skip any blank line
while lines[line_idx].strip() == '':
    line_idx += 1

# Parse depot coordinates
depots_coord = []
while len(depots_coord) < num_depots:
    line = lines[line_idx].strip()
    if line:
        depots_coord.append(parse_numbers(line))
    line_idx += 1

# Parse customer coordinates
customers_coord = []
while len(customers_coord) < num_customers:
    line = lines[line_idx].strip()
    if line:
        customers_coord.append(parse_numbers(line))
    line_idx += 1

# Parse vehicle capacity
while lines[line_idx].strip() == '':
    line_idx += 1
vehicle_capacity = int(lines[line_idx].strip()); line_idx += 1

# Parse depot capacities
depot_capacities, line_idx = collect_values(lines, line_idx, num_depots)

# Parse customer demands
customer_demands, line_idx = collect_values(lines, line_idx, num_customers)

# Parse depot opening costs
depot_opening_costs, line_idx = collect_values(lines, line_idx, num_depots)

# Parse route opening cost
while lines[line_idx].strip() == '':
    line_idx += 1
route_opening_cost = float(lines[line_idx].strip()); line_idx += 1

# Parse cost type (0 or 1)
while line_idx < len(lines) and lines[line_idx].strip() == '':
    line_idx += 1
cost_type = int(lines[line_idx].strip()); line_idx += 1

# Store all parsed data
instance_1112_LLRP = {
    "num_customers": num_customers,
    "num_depots": num_depots,
    "depots_coord": depots_coord,
    "customers_coord": customers_coord,
    "vehicle_capacity": vehicle_capacity,
    "depot_capacities": depot_capacities,
    "customer_demands": customer_demands,
    "depot_opening_costs": depot_opening_costs,
    "route_opening_cost": route_opening_cost,
    "cost_type": cost_type
}

print(instance_1112_LLRP)

import matplotlib.pyplot as plt
import webbrowser

def instance_1112_LLRP_nodes(instance_1112_LLRP, save_path="llrp_plot.png"):
    depots = instance_1112_LLRP["depots_coord"]
    customers = instance_1112_LLRP["customers_coord"]

    depot_x = [d[0] for d in depots]
    depot_y = [d[1] for d in depots]
    customer_x = [c[0] for c in customers]
    customer_y = [c[1] for c in customers]

    plt.figure(figsize=(10, 8))
    plt.scatter(depot_x, depot_y, c='red', marker='s', s=100, label='Depots')
    for i, (x, y) in enumerate(depots):
        plt.text(x + 1, y + 1, f"D{i}", color='red', fontsize=9)

    plt.scatter(customer_x, customer_y, c='blue', marker='o', s=40, label='Customers')
    for i, (x, y) in enumerate(customers):
        plt.text(x + 0.5, y + 0.5, f"C{i}", color='blue', fontsize=7)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Instance 1112 LLRP: Depot and Customer Nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and auto-open
    plt.savefig(save_path)
    plt.close()
    print(r"Plot saved to C:\Users\pengr\Desktop\DP_material\TB_data_exp\instance_plot.png")
    webbrowser.open(r"C:\Users\pengr\Desktop\DP_material\TB_data_exp\instance_plot.png")


