# Decision Modeling Assignment 2

**Name:** Oluwanifemi Olajuyigbe

## Overview
This project contains two linear programming optimization problems implemented using Python's PuLP library.

## Files

### 1. Vacation Bag Problem
- **File:** `Decision Modeling_Vacation_bag_Oluwanifemi_Olajuyigbe.py`
- **Objective:** Maximize value while respecting weight constraints
- **Scenarios:** Tests with 20kg, 23kg, and 26kg weight limits

### 2. Paris Visit Problem  
- **File:** `Decision_Modeling_Assignment_Paris_Visit_Oluwanifemi_Olajuyigbe.py`
- **Problem:** Tourist site selection optimization
- **Objective:** Maximize number of sites visited within budget and time constraints
- **Features:**
  - Multiple budget/duration scenarios
  - Preference constraints (must-visit sites, conditional logic)
  - Ranking and Statistical correlation analysis between duration, appreciation, and price

### 3. Results
- **File:** `results.txt` - Output from running the Paris visit optimization

## Requirements
- Python 3.x
- PuLP (linear programming library)
- SciPy (for statistical analysis)

## Usage
Run the Python files directly:
```bash
python "Decision Modeling_Vacation_bag_Oluwanifemi_Olajuyigbe.py"
python "Decision_Modeling_Assignment_Paris_Visit_Oluwanifemi_Olajuyigbe.py"
```

## Key Results
- **Vacation Bag:** Optimal packing solutions for different weight limits
- **Paris Visit:** Site recommendations based on various constraints and preferences
