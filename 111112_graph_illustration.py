# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:46:37 2025

@author: pengr
"""

import matplotlib.pyplot as plt
import numpy as np

# === Load and clean file ===
file_path = "C:\\Users\\pengr\\OneDrive\\Desktop\\DP_material\\TB_data\\coordP111112.dat"


with open(file_path, "r") as file:
    lines = [line.strip() for line in file if line.strip()]

# === Parse basic parameters ===
num_customers = int(lines[0])
num_depots = int(lines[1])

# === Parse coordinates ===
# Depot coordinates
depot_coords = []
for i in range(2, 2 + num_depots):
    x, y = map(float, lines[i].split())
    depot_coords.append((x, y))

# Customer coordinates
customer_coords = []
customer_start = 2 + num_depots
for i in range(customer_start, customer_start + num_customers):
    x, y = map(float, lines[i].split())
    customer_coords.append((x, y))

# === Parse other parameters ===
vehicle_capacity = int(lines[customer_start + num_customers])
depot_capacities = list(map(int, lines[customer_start + num_customers + 1 : customer_start + num_customers + 1 + num_depots]))
customer_demands = list(map(int, lines[customer_start + num_customers + 1 + num_depots : customer_start + num_customers + 1 + num_depots + num_customers]))
depot_opening_costs = list(map(int, lines[customer_start + num_customers + 1 + num_depots + num_customers : customer_start + num_customers + 1 + num_depots + num_customers + num_depots]))
route_cost = float(lines[customer_start + num_customers + 1 + num_depots + num_customers + num_depots])
cost_type = int(lines[customer_start + num_customers + 1 + num_depots + num_customers + num_depots + 1])

# === Convert to NumPy arrays ===
depot_coords = np.array(depot_coords)
customer_coords = np.array(customer_coords)

# === Plot the instance ===
plt.figure(figsize=(10, 10))
plt.scatter(customer_coords[:, 0], customer_coords[:, 1], c='blue', label='Customers', s=30)
plt.scatter(depot_coords[:, 0], depot_coords[:, 1], c='red', marker='s', label='Depots', s=100)

# Label depots and some customers for clarity
for i, (x, y) in enumerate(depot_coords):
    plt.text(x + 0.5, y, f'D{i}', color='darkred')
for i, (x, y) in enumerate(customer_coords):
    if i % 10 == 0:
        plt.text(x + 0.5, y, f'C{i}', fontsize=8, color='darkblue')

plt.title('Instance 111112 - Location Routing Graph (Exact Data)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
