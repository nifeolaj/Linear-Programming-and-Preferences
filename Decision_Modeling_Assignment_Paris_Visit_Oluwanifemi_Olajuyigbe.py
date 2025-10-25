from pulp import *
from scipy.stats import kendalltau, spearmanr

sites = ['TE', 'ML', 'AT', 'MO', 'JT', 'CA', 'CP', 'CN', 'BS', 'SC', 'PC', 'TM', 'AC']
duration = {'TE':4.5, 'ML': 3, 'AT':1, 'MO':2, 'JT':1.5, 'CA':2, 'CP':2.5, 'CN':2, 'BS':2, 'SC':1.5, 'PC':0.75, 'TM':2, 'AC':1.5}
appreciations = {'TE':5, 'ML':4, 'AT':3, 'MO':2, 'JT':3, 'CA':4, 'CP':1, 'CN':5, 'BS':4, 'SC':1, 'PC':3, 'TM':2, 'AC':5}
price = {'TE':16.5, 'ML':14, 'AT':10.5, 'MO':11, 'JT':0, 'CA':10, 'CP':10, 'CN':7, 'BS':10, 'SC':8.5, 'PC':0, 'TM':12, 'AC':0}
distances = {
    'TE': {'TE': 0, 'ML': 3.8, 'AT': 2.1, 'MO': 2.4, 'JT': 3.5, 'CA': 4.2, 'CP': 5.0, 'CN': 4.4, 'BS': 5.5, 'SC': 4.2, 'PC': 2.5, 'TM': 3.1, 'AC': 1.9},
    'ML': {'TE': 3.8, 'ML': 0, 'AT': 3.8, 'MO': 1.1, 'JT': 1.3, 'CA': 3.3, 'CP': 1.3, 'CN': 1.1, 'BS': 3.4, 'SC': 0.8, 'PC': 1.7, 'TM': 2.5, 'AC': 2.8},
    'AT': {'TE': 2.1, 'ML': 3.8, 'AT': 0, 'MO': 3.1, 'JT': 3.0, 'CA': 5.8, 'CP': 4.8, 'CN': 4.9, 'BS': 4.3, 'SC': 4.6, 'PC': 2.2, 'TM': 4.4, 'AC': 1.0},
    'MO': {'TE': 2.4, 'ML': 1.1, 'AT': 3.1, 'MO': 0, 'JT': 0.9, 'CA': 3.1, 'CP': 2.5, 'CN': 2.0, 'BS': 3.9, 'SC': 1.8, 'PC': 1.0, 'TM': 2.3, 'AC': 2.1},
    'JT': {'TE': 3.5, 'ML': 1.3, 'AT': 3.0, 'MO': 0.9, 'JT': 0, 'CA': 4.2, 'CP': 2.0, 'CN': 2.4, 'BS': 2.7, 'SC': 2.0, 'PC': 1.0, 'TM': 3.4, 'AC': 2.1},
    'CA': {'TE': 4.2, 'ML': 3.3, 'AT': 5.8, 'MO': 3.1, 'JT': 4.2, 'CA': 0, 'CP': 3.5, 'CN': 2.7, 'BS': 6.5, 'SC': 2.6, 'PC': 3.8, 'TM': 1.3, 'AC': 4.9},
    'CP': {'TE': 5.0, 'ML': 1.3, 'AT': 4.8, 'MO': 2.5, 'JT': 2.0, 'CA': 3.5, 'CP': 0, 'CN': 0.85, 'BS': 3.7, 'SC': 0.9, 'PC': 2.7, 'TM': 3.4, 'AC': 3.8},
    'CN': {'TE': 4.4, 'ML': 1.1, 'AT': 4.9, 'MO': 2.0, 'JT': 2.4, 'CA': 2.7, 'CP': 0.85, 'CN': 0, 'BS': 4.5, 'SC': 0.4, 'PC': 2.8, 'TM': 2.7, 'AC': 3.9},
    'BS': {'TE': 5.5, 'ML': 3.4, 'AT': 4.3, 'MO': 3.9, 'JT': 2.7, 'CA': 6.5, 'CP': 3.7, 'CN': 4.5, 'BS': 0, 'SC': 4.2, 'PC': 3.3, 'TM': 5.7, 'AC': 3.8},
    'SC': {'TE': 4.2, 'ML': 0.8, 'AT': 4.6, 'MO': 1.8, 'JT': 2.0, 'CA': 2.6, 'CP': 0.9, 'CN': 0.4, 'BS': 4.2, 'SC': 0, 'PC': 2.5, 'TM': 2.6, 'AC': 3.6},
    'PC': {'TE': 2.5, 'ML': 1.7, 'AT': 2.2, 'MO': 1.0, 'JT': 1.0, 'CA': 3.8, 'CP': 2.7, 'CN': 2.8, 'BS': 3.3, 'SC': 2.5, 'PC': 0, 'TM': 3.0, 'AC': 1.2},
    'TM': {'TE': 3.1, 'ML': 2.5, 'AT': 4.4, 'MO': 2.3, 'JT': 3.4, 'CA': 1.3, 'CP': 3.4, 'CN': 2.7, 'BS': 5.7, 'SC': 2.6, 'PC': 3.0, 'TM': 0, 'AC': 2.1},
    'AC': {'TE': 1.9, 'ML': 2.8, 'AT': 1.0, 'MO': 2.1, 'JT': 2.1, 'CA': 4.9, 'CP': 3.8, 'CN': 3.9, 'BS': 3.8, 'SC': 3.6, 'PC': 1.2, 'TM': 2.1, 'AC': 0}
}

def paris_visit(max_duration, budget):
    prob = LpProblem("ListVisit 1", LpMaximize)
    
    # Create decision variables
    x = LpVariable.dicts("site", sites, 0, 1, cat="Binary")

    # Objective function
    prob += lpSum(x[i] for i in sites), "Total Sites Visited"

    # Constraints
    prob += lpSum(duration[i] * x[i] for i in sites) <= max_duration, "Total Duration"
    prob += lpSum(price[i] * x[i] for i in sites) <= budget, "Total Budget"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    print("Status:", LpStatus[prob.status])

    chosen_places = [i for i in sites if x[i].value() == 1]
    number_of_sites = len(chosen_places)
    total_duration = sum(duration[i] for i in chosen_places)
    total_amount= sum(price[i] for i in chosen_places)

    print(f"Max Budget = €{budget}, Max Duration = {max_duration} hours")
    print(f"Number of sites: {number_of_sites}")
    print(f"Chosen Objects: {chosen_places}")
    print(f"Total amount = €{total_amount}, Total duration = {total_duration} hours")
    print("-"*90)

    return chosen_places

# (a) Which list(s) of places could you recommend to him?
print("(1a) Which list(s) of places could you recommend to him?")
ListVisit_1 = paris_visit(14, 75)
# Max Budget = €75, Max Duration = 14 hours
# Chosen Objects: ['AT', 'MO', 'JT', 'CN', 'BS', 'SC', 'PC', 'AC']
# Total amount = €47.0, Total duration = 12.25 hours       


# (b) Which list(s) of places could you recommend to him if Doe’s budget is now 65 e, the maximum duration still being 14 hours ?
print("(1b) Which list(s) of places could you recommend to him if Doe’s budget is now 65 e, the maximum duration still being 14 hours ?")
paris_visit(14, 65)
# Max Budget = €65, Max Duration = 14 hours
# Chosen Objects: ['AT', 'MO', 'JT', 'CN', 'BS', 'SC', 'PC', 'AC']
# Total amount = €47.0, Total duration = 12.25 hours   

# (c) Which list(s) of places could you recommend to him if Doe’s budget is now 90 e and the maximum duration being 10 hours ?
print("(1c) Which list(s) of places could you recommend to him if Doe’s budget is now 90 e and the maximum duration being 10 hours ?")
paris_visit(10, 90)
# Max Budget = €90, Max Duration = 10 hours
# Chosen Objects: ['AT', 'MO', 'CN', 'SC', 'PC', 'AC']     
# Total amount = €37.0, Total duration = 8.75 hours 


#----------------------------------------------------------------------------------------------------------------------------------------------#
# 2. PREFERRENCES
#----------------------------------------------------------------------------------------------------------------------------------------------#
print("\n" + "="*90)
print("2. PREFERRENCES")
print("="*90)
def visit_preferences(preferences, max_duration=14, budget=75):
    prob = LpProblem("ListVisit 2", LpMaximize)

    # Create decision variables
    x = LpVariable.dicts("site", sites, 0, 1, cat="Binary")

    # Objective function
    prob += lpSum(x[i] for i in sites), "Total Sites Visited"

    # Constraints
    prob += lpSum(duration[i] * x[i] for i in sites) <= max_duration, "Total Duration"
    prob += lpSum(price[i] * x[i] for i in sites) <= budget, "Total Budget"

    # Preference Constraints
    if 'close_sites' in preferences:
        for i, (site1, site2) in enumerate(preferences['close_sites']):
            # Both should be visited together or not at all 
            prob += x[site1] == x[site2], f"Close_Sites_{site1}_{site2}_{i}"
            
    if 'must_visit' in preferences:
        for i, site in enumerate(preferences['must_visit']):
            # Must visit the site 
            prob += x[site] == 1, f"Must_Visit_{site}_{i}"
            
    if 'if_then' in preferences:
        for i, (site1, site2) in enumerate(preferences['if_then']):
            # If site1 is visited, then site2 must be visited 
            prob += x[site2] >= x[site1], f"If_Then_{site1}_{site2}_{i}"
            
    if 'if_then_not' in preferences:
        for i, (site1, site2) in enumerate(preferences['if_then_not']):
            # If site1 is visited, then site2 must not be visited 
            prob += x[site1] + x[site2] <= 1, f"If_Then_Not_{site1}_{site2}_{i}"
            
    
    
    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    print("Status:", LpStatus[prob.status])

    chosen_sites = [i for i in sites if x[i].value() == 1]
    number_of_sites = len(chosen_sites)
    total_duration = sum(duration[i] for i in chosen_sites)
    total_amount = sum(price[i] for i in chosen_sites)

    print(f"Max Budget = €{budget}, Max Duration = {max_duration} hours")
    print(f"Number of sites: {number_of_sites}")
    print(f"Chosen Sites: {chosen_sites}")
    print(f"Total amount = €{total_amount}, Total duration = {total_duration} hours")

    return chosen_sites

# function to find close sites
def find_close_sites(max_distance=1):
    close_sites = []
    for i, site1 in enumerate(sites):
        for site2 in sites[i+1:]:  # Only check sites after site1 to avoid duplicates
            if distances[site1][site2] <= max_distance:
                close_sites.append((site1, site2))
    return close_sites

# function to check if two lists are identical
def is_identical(list1, list2):
    return sorted(list1) == sorted(list2)


close_sites = find_close_sites(1)
print(f"Close sites: {close_sites}")

# 2(a) - Individual Preferences:
print("-"*90)
print("(2a) - Individual Preferences:")
# Preference 1: Close sites
print("\nPreference 1: Close sites")
pref1 = {'close_sites': close_sites}
pref1_list = visit_preferences(pref1)
print(f"Different from ListVisit_1? {not is_identical(pref1_list, ListVisit_1)}")
print("-"*90)

# Preference 2: Must visit TE and CA
print("\nPreference 2: Must visit TE and CA")
pref2 = {'must_visit': ['TE', 'CA']}
pref2_list = visit_preferences(pref2)
print(f"Different from ListVisit_1? {not is_identical(pref2_list, ListVisit_1)}")
print("-"*90)
# Preference 3: If AC then not SC
print("\nPreference 3: If AC then not SC")
pref3 = {'if_then_not': [('AC', 'SC')]}
pref3_list = visit_preferences(pref3)
print(f"Different from ListVisit_1? {not is_identical(pref3_list, ListVisit_1)}")
print("-"*90)
# Preference 4: Must visit AT
print("\nPreference 4: Must visit AT")
pref4 = {'must_visit': ['AT']}
pref4_list = visit_preferences(pref4)
print(f"Different from ListVisit_1? {not is_identical(pref4_list, ListVisit_1)}")
print("-"*90)   
# Preference 5: If ML then MO
print("\nPreference 5: If ML then MO")
pref5 = {'if_then': [('ML', 'MO')]}
pref5_list = visit_preferences(pref5)
print(f"Different from ListVisit_1? {not is_identical(pref5_list, ListVisit_1)}")
print("-"*90)

# 2(b) - Preference 1 AND 2
print("(2b) - Preference 1 AND 2")
pref1_2 = {'close_sites': close_sites, 'must_visit': ['TE', 'CA']}
visit_preferences(pref1_2)
print("-"*90)

# 2(c) - Preference 1 AND 3
print("(2c) - Preference 1 AND 3")
pref1_3 = {'close_sites': close_sites, 'if_then_not': [('AC', 'SC')]}
visit_preferences(pref1_3)
print("-"*90)

# 2(d) - Preference 1 AND 4
print("(2d) - Preference 1 AND 4")
pref1_4 = {'close_sites': close_sites, 'must_visit': ['AT']}
visit_preferences(pref1_4)
print("-"*90)

# 2(e) - Preference 2 AND 5
print("(2e) - Preference 2 AND 5")
pref2_5 = {'must_visit': ['TE', 'CA'], 'if_then': [('ML', 'MO')]}
visit_preferences(pref2_5)
print("-"*90)

# 2(f) - Preference 3 AND 4
print("(2f) - Preference 3 AND 4")
pref3_4 = {'if_then_not': [('AC', 'SC')], 'must_visit': ['AT']}
visit_preferences(pref3_4)
print("-"*90)

# 2(g) - Preference 4 AND 5
print("(2g) - Preference 4 AND 5")
pref4_5 = {'must_visit': ['AT'], 'if_then': [('ML', 'MO')]}
visit_preferences(pref4_5)
print("-"*90)

# 2(h) - Preference 1 AND 2 AND 4
print("(2h) - Preference 1 AND 2 AND 4")
pref1_2_4 = {'close_sites': close_sites, 'must_visit': ['TE', 'CA', 'AT']}
visit_preferences(pref1_2_4)
print("-"*90)

# 2(i) - Preference 2 AND 3 AND 5
print("(2i) - Preference 2 AND 3 AND 5")
pref2_3_5 = {'must_visit': ['TE', 'CA'], 'if_then_not': [('AC', 'SC')], 'if_then': [('ML', 'MO')]}
visit_preferences(pref2_3_5)
print("-"*90)

# 2(j) - Preference 2 AND 3 AND 4 AND 5
print("(2j) - Preference 2 AND 3 AND 4 AND 5")
pref2_3_4_5 = {'must_visit': ['TE', 'CA', 'AT'], 'if_then_not': [('AC', 'SC')], 'if_then': [('ML', 'MO')]}
visit_preferences(pref2_3_4_5)
print("-"*90)

# 2(k) - Preference 1 AND 2 AND 4 AND 5
print("(2k) - Preference 1 AND 2 AND 4 AND 5")
pref1_2_4_5 = {'close_sites': close_sites, 'must_visit': ['TE', 'CA', 'AT'], 'if_then': [('ML', 'MO')]}
visit_preferences(pref1_2_4_5)
print("-"*90)

# 2(l) - Preference 1 AND 2 AND 3 AND 4 AND 5
print("(2l) - Preference 1 AND 2 AND 3 AND 4 AND 5")
pref_1_2_3_4_5 = {'close_sites': close_sites, 'must_visit': ['TE', 'CA', 'AT'], 'if_then_not': [('AC', 'SC')], 'if_then': [('ML', 'MO')]}
visit_preferences(pref_1_2_3_4_5)
print("-"*90)

# ----------------------------------------------------------------------------------------------------------------------------------------------#
# 3: Rankings 
# ----------------------------------------------------------------------------------------------------------------------------------------------#
print("\n" + "="*90)
print("3. RANKINGS")
print("="*90)

duration_ranking = sorted(sites, key=lambda s: duration[s]) # minimize
appreciation_ranking = sorted(sites, key=lambda s: appreciations[s], reverse=True) # maximize
price_ranking = sorted(sites, key=lambda s: price[s]) # minimize

print(f"\nDuration Ranking: {duration_ranking}")
print(f"\nAppreciation Ranking: {appreciation_ranking}")
print(f"\nPrice Ranking: {price_ranking}")


# Convert to numeric ranks for correlation
dur_ranks = {site: i+1 for i, site in enumerate(duration_ranking)}
app_ranks = {site: i+1 for i, site in enumerate(appreciation_ranking)}
pri_ranks = {site: i+1 for i, site in enumerate(price_ranking)}

# Create arrays for correlation
dur_array = [dur_ranks[s] for s in sites]
app_array = [app_ranks[s] for s in sites]
pri_array = [pri_ranks[s] for s in sites]

# Calculate correlations
kendall_dur_app, p_kendall_da = kendalltau(dur_array, app_array)
kendall_dur_pri, p_kendall_dp = kendalltau(dur_array, pri_array)
kendall_app_pri, p_kendall_ap = kendalltau(app_array, pri_array)

spearman_dur_app, p_spearman_da = spearmanr(dur_array, app_array)
spearman_dur_pri, p_spearman_dp = spearmanr(dur_array, pri_array)
spearman_app_pri, p_spearman_ap = spearmanr(app_array, pri_array)

print("\n" + "="*70)

print("\nKendall Tau Correlation:")
print(f"Duration vs Appreciation: Kendall = {kendall_dur_app:.4f} (p = {p_kendall_da:.4f})")
print(f"Duration vs Price: Kendall = {kendall_dur_pri:.4f} (p = {p_kendall_dp:.4f})")
print(f"Appreciation vs Price: Kendall = {kendall_app_pri:.4f} (p = {p_kendall_ap:.4f})")

print("\nSpearman Rank Correlation:")
print(f"Duration vs Appreciation: Spearman = {spearman_dur_app:.4f} (p = {p_spearman_da:.4f})")
print(f"Duration vs Price: Spearman = {spearman_dur_pri:.4f} (p = {p_spearman_dp:.4f})")
print(f"Appreciation vs Price: Spearman = {spearman_app_pri:.4f} (p = {p_spearman_ap:.4f})")
print("\n" + "="*70)
print("""
Based on the Kendall's Tau and Spearman's Rank correlation results, the three rankings for Duration, Appreciation, and Price are different from each other.
\nThe only clear relationship is between Duration and Price, which shows a positive and statistically significant correlation.
This makes sense as sites that take longer to visit generally cost more.
\nHowever, looking at Duration vs Appreciation and Appreciation vs Price, the correlations are weak and not statistically significant (p > 0.05). 
\nThis shows that visitors appreciation of a site has nothing to do with how long it takes to visit or how much it costs.
\nHence, the three rankings are quite different with the only real similarity being between Duration and Price. 
""")
