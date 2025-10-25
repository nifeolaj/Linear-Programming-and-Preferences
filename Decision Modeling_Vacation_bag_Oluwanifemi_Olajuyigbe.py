# Vacation Objects

from pulp import *


objects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
values = {'A': 10, 'B': 8, 'C': 12, 'D': 4, 'E': 5, 'F': 10, 'G': 6, 'H': 9, 'I': 7, 'J': 9}
weights = {'A': 5, 'B': 7, 'C': 4, 'D': 3, 'E': 5, 'F': 3, 'G': 4, 'H': 6, 'I': 4, 'J': 6}

def vacation_bag(max_weight):
    prob = LpProblem("Vacation Bag", LpMaximize)

    # Create decision variables
    x = LpVariable.dicts("Object", objects, 0, 1, cat="Binary")

    # Objective function
    prob += lpSum(values[i] * x[i] for i in objects), "Total Value"

    # Constraint
    prob += lpSum(weights[i] * x[i] for i in objects) <= max_weight, "Total Weight"
    
    # Solve the problem
    prob.solve()

    print("Status:", LpStatus[prob.status])

    chosen_objects = [i for i in objects if x[i].value() == 1]
    total_value = sum(values[i] for i in chosen_objects)
    total_weight = sum(weights[i] for i in chosen_objects)

    print(f"Max Weight = {max_weight} kg")
    print(f"Chosen Objects: {chosen_objects}")
    print(f"Total value = €{total_value}, Total weight = {total_weight} kg")
    print("-"*40)


# Which items should Chris put in the bag
vacation_bag(23)

# Which solution we obtain if the maximum weight is now 20 kgs ?
vacation_bag(20)

# Which solution we obtain if the maximum weight is now 26 kgs ?
vacation_bag(26)