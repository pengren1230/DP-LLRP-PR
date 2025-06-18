# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:07:52 2025

@author: pengr
"""

import matplotlib.pyplot as plt

# Re-defining the coordinates after kernel reset
depot_coords = [(6, 7), (19, 44), (37, 23), (35, 6), (5, 8)]
customer_coords = [
    (20, 35), (8, 31), (29, 43), (18, 39), (19, 47), (31, 24), (38, 50),
    (33, 21), (2, 27), (1, 12), (26, 20), (20, 33), (15, 46), (20, 26),
    (17, 19), (15, 12), (5, 30), (13, 40), (38, 5), (9, 40)
]

# Plotting
plt.figure(figsize=(10, 8))

# Plot depots
for i, (x, y) in enumerate(depot_coords):
    plt.scatter(x, y, c='red', marker='s', s=100, label='Depot' if i == 0 else "")
    plt.text(x + 0.5, y, f'P{i+1}', fontsize=9, color='red')

# Plot customers
for i, (x, y) in enumerate(customer_coords):
    plt.scatter(x, y, c='blue', marker='o', s=60, label='Customer' if i == 0 else "")
    plt.text(x + 0.5, y, f'I{i+1}', fontsize=8, color='blue')

plt.title('LLRP Problem Setup: Depots and Customers')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()