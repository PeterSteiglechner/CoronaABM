"""
This script provides functions that create
- a net of households.
    A household is a (fully-connected) cluster of 1, 2, 3, 4, or 5 people.
    High Risk individuals only appear in households of 1 or 2 people.
    Children appear only in households of >2 people with at least one adult.
    Required: A given distribution of sizes of households.
- a net of friends/evening partners.
    required: correlation matrix between the groups.

- a random network
    required: each agent's node degree, i.e. Nr of random contacts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx

N_AVERAGE_FRIENDS = [4, 3, 3]
# CORRELATIONS_FRIENDS = [[0.6, 0.3, 0.1],  # children
#                          [0.3, 0.4, 0.3],  # adults
#                          [0.1, 0.3, 0.6]  # High risk
#                          ]
CORRELATIONS_FRIENDS = [[0.9, 0.01, 0.0],  # children
                        [0.01, 0.9, 0.09],  # adults
                        [0.0, 0.09, 0.91]  # High risk
                        ]

N_AVERAGE_WORK = [6, 4, 2]
CORRELATIONS_WORK = [[0.9, 0.1, 0.0],  # children
                        [0.1, 0.8, 0.1],  # adults
                        [0.0, 0.1, 0.9]  # High risk
                        ]

# Fractions of 1-, 2-, 3-, 4-, 5-person households
# For Germany: https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/lrbev05.html
HOUSEHOLD_FRACTIONS = np.array([42.3, 33.2, 11.9, 9.1, 3.5])



AVG_RANDOM_CONTACTS = [3, 2, 1]
# Plot the distribution of random contacts in the three groups as:
# bins = np.arange(-0.5, 15.5, step=1)
# for k in range(3):
#      nr_random_contacts = [ag.randomContacts for ag in agents[indices_groups[k][0]:indices_groups[k][-1]]]
#     plt.hist(nr_random_contacts, bins = bins, alpha = 0.5)
# plt.show()


def create_Household_Network(agents, N_subgroups, HOUSEHOLD_FRACTIONS, edge_weight):
    """
    Households
    """
    indices_groups = [
        np.arange(N_subgroups[0]),  # children
        np.sum(N_subgroups[:1]) + np.arange(N_subgroups[1]),  # adults
        np.sum(N_subgroups[:2]) + np.arange(N_subgroups[2])  # risk
    ]

    N_agents = len(agents)
    N_HH = N_agents / np.dot(HOUSEHOLD_FRACTIONS, np.arange(5) + 1)
    N_HHs = np.array(HOUSEHOLD_FRACTIONS * N_HH)
    N_HHs = np.ceil(N_HHs).astype(int)
    HH_occ = dict(
        zip(
            np.arange(N_HHs.sum()),  # Index of Household
            np.zeros(N_HHs.sum())))  # =0, initially all households empty
    HH_caps = dict(
        zip(
            np.arange(N_HHs.sum()),  # Index
            np.concatenate([[i + 1 for _ in range(N_HHs[i])] for i in range(len(N_HHs))])  # and Capcity of Household
        )
    )

    family_net = np.zeros([N_agents, N_agents])

    hhs = list(HH_caps.keys())
    # Risk Group:
    a = np.where(np.array(list(HH_caps.values())) <= 2)[0]
    allowed_risk = list(np.array(hhs)[a])
    full_groups = []
    for i in indices_groups[2]:
        ag = agents[i]
        ag.hh = hhs[np.random.choice(allowed_risk)]
        HH_occ[ag.hh] += 1
        if HH_occ[ag.hh] >= HH_caps[ag.hh]:
            allowed_risk.remove(ag.hh)
            full_groups.append(ag.hh)
            # hhs.remove(ag.hh)

    # Children:
    a = np.where(np.array(list(HH_caps.values())) > 2)[0]
    allowed_children = list(np.array(hhs)[a])
    for i in indices_groups[0]:
        ag = agents[i]
        ag.hh = hhs[np.random.choice(allowed_children)]
        HH_occ[ag.hh] += 1
        if HH_occ[ag.hh] == HH_caps[ag.hh] - 1:  # SPACE FOR AT LEAST ONE ADULT
            allowed_children.remove(ag.hh)
            # hhs.remove(ag.hh) # parents still allowed

    # adults:
    for hh in full_groups:
        hhs.remove(hh)

    for i in indices_groups[1]:
        ag = agents[i]
        ag.hh = np.random.choice(hhs)
        HH_occ[ag.hh] += 1
        if HH_occ[ag.hh] == HH_caps[ag.hh]:  # SPACE FOR AT LEAST ONE ADULT
            hhs.remove(ag.hh)

    # Loop through Households and add 1s for those people belonging to the household.
    hhs_of_agents = np.array([ag.hh for ag in agents])
    all_hhs = np.unique(hhs_of_agents)
    for hh in all_hhs:
        ags = np.where(hhs_of_agents == hh)[0]
        for i in ags:
            for j in ags:
                family_net[i, j] = 1
    np.fill_diagonal(family_net, 0)     # set diagonal to 0.

    H = nx.from_numpy_matrix(edge_weight*family_net)    # Create Network!

    return H

def create_Friends_Network(agents, N_subgroups, N_AVERAGE_FRIENDS, CORRELATIONS_FRIENDS, edge_weight):
    """
    FRIENDS
    """
    indices_groups = [
        np.arange(N_subgroups[0]),                              # children
        np.sum(N_subgroups[:1]) + np.arange(N_subgroups[1]),    # adults
        np.sum(N_subgroups[:2]) + np.arange(N_subgroups[2])     # risk
    ]
    N_agents = len(agents)
    freetime_net = np.zeros([N_agents, N_agents])
    N_subgroups = [len(ags) for ags in indices_groups]
    for k in range(len(indices_groups)):
        for j in range(len(indices_groups)):
            p_sub = min(CORRELATIONS_FRIENDS[k][j] * N_AVERAGE_FRIENDS[k] / N_subgroups[k], 1)
            subnet = np.random.choice([0, 1], size=(N_subgroups[k], N_subgroups[j]), p=[1 - p_sub, p_sub])
            if k == j:
                upper_triangle_indices = np.triu_indices(N_subgroups[k])
                subnet[upper_triangle_indices] = subnet.T[upper_triangle_indices]
                np.fill_diagonal(subnet, 0)
            a, b = indices_groups[k], indices_groups[j]
            freetime_net[a[0]:a[-1] + 1, b[0]:b[-1] + 1] = subnet

    F = nx.from_numpy_matrix(edge_weight * freetime_net, parallel_edges=False)

    return F

"""
DayActivity
"""

def create_Random_Network(agents, N_subgroups, AVG_RANDOM_CONTACTS, edge_weight):
    """
    Random
    """
    #random_net = np.zeros([N_AGENTS, N_AGENTS])

    c = 0  # Counter variable for break-condition
    while not nx.is_graphical([ag.randomContacts for ag in agents]):
        # If True: the chosen nr of random contacts (i.e. node degrees) can not be translated into a network.
        # Hence, choose a new sequence of node degrees (from same distribution) until the sequence is graphical
        for ag in agents:
            ag.randomContacts = stats.poisson(AVG_RANDOM_CONTACTS[ag.group]).rvs()
        c += 1
        if c > 1000:
            # This is for safety, i.e. not getting stuck in the While Loop.
            print("Error: Can't find graphical random contact sequence")
            break

    # Previous Solution: Create Random Matrix BY HAND
    # for i, ag in enumerate(agents):
    #    already_contact = random_net[:, i].sum()
    #    if 0 < ag.randomContacts - already_contact < N_AGENTS - i:
    #        contacts = np.random.choice(agents[i+1:], size=np.int(ag.randomContacts - already_contact))
    #        for contact in contacts:
    #            j = contact.id
    #            random_net[i, j] = 1
    #            random_net[j, i] = 1

    # Create random network from a sequence of node degrees specified for each agent
    r = nx.random_degree_sequence_graph([ag.randomContacts for ag in agents])
    # Create the adjacency Matrix from that graph
    random_net = nx.adjacency_matrix(r)
    R = nx.from_numpy_matrix(edge_weight * random_net.todense(), parallel_edges=False)
    return R


def create_WattsStogarts_Network(agents, indices_groups, k, p):
    w = nx.watts_strogatz_graph(len(agents), k, p)
    return w


