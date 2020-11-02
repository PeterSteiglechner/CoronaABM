import numpy as np

# class policy:
#     t_start = np.nan
#     t_end = np.nan
#
# def initiate_policy(t_start, policy):
#     if t>t_start:
#         for ag in agents:
#             ag.policies.extend("isolate_sick")
#
#
# def isolate_sick(ag, contact_lists):
#     if ag.state == "i_s":
#         contact_lists =


def policy_0(ag, agents, t, all_contacts):
    """ Default """
    household_contacts, random_contacts, activity_contacts, work_contacts = all_contacts
    contacts = []
    ampl_factors = []
    for c in [household_contacts, random_contacts, activity_contacts, work_contacts]:
        contacts.extend(c)
        ampl_factors.extend([1])
    return contacts, ampl_factors


def policy_1(ag, agents, t, all_contacts):
    """ Quarantine all symptomatic agents and social distance for all first contacts"""
    household_contacts, random_contacts, activity_contacts, work_contacts = all_contacts

    if ag.state == "i_s":
        ag.isolation_status[0] = "quarantine"
        for c_group in [household_contacts, activity_contacts, work_contacts]:
            for c in c_group:
                if not agents[c].isolation_status[0] == "quarantine":
                    agents[c].isolation_status[0] = "social distance"
                    agents[c].isolation_status[1] = t+14   # End time of social distance
    contacts = []
    ampl_factors = []
    if ag.isolation_status[0] == "no isolation":
        for c in [household_contacts, random_contacts, activity_contacts, work_contacts]:
            contacts.extend(c)
            ampl_factors.extend([1])
    elif ag.isolation_status[0] == "social distance":
        for c in [household_contacts]:
            contacts.extend(c)
            ampl_factors.extend([1])
        for c in [random_contacts, work_contacts]:
            contacts.extend(c)
            ampl_factors.extend([0.5])
    elif ag.isolation_status[0] == "quarantine":
        for c in [household_contacts]:
            contacts.extend(c)
            ampl_factors.extend([0.5])
    return contacts, ampl_factors


def policy_2(ag, agents, t, all_contacts):
    """ Mask on top of quarantine symptoms and distance their neighbours etc"""
    household_contacts, random_contacts, activity_contacts, work_contacts = all_contacts

    if ag.state == "i_s":
        ag.isolation_status[0] = "quarantine"
        for c_group in [household_contacts, activity_contacts, work_contacts]:
            for c in c_group:
                if not agents[c].isolation_status[0] == "quarantine":
                    agents[c].isolation_status[0] = "social distance"
                    agents[c].isolation_status[1] = t+14   # End time of social distance

    masc_factor = 0.5

    contacts = []
    ampl_factors = []
    if ag.isolation_status[0] == "no isolation":
        for c in [household_contacts]:
            contacts.extend(c)
            ampl_factors.extend([1])
        for c in [random_contacts, activity_contacts, work_contacts]:
            contacts.extend(c)
            ampl_factors.extend([1*masc_factor])
    elif ag.isolation_status[0] == "social distance":
        for c in [household_contacts]:
            contacts.extend(c)
            ampl_factors.extend([1])
        for c in [random_contacts, work_contacts]:
            contacts.extend(c)
            ampl_factors.extend([min(0.5, masc_factor)])
    elif ag.isolation_status[0] == "quarantine":
        for c in [household_contacts]:
            contacts.extend(c)
            ampl_factors.extend([0.5])
    return contacts, ampl_factors
