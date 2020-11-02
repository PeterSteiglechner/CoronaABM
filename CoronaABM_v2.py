#!/usr/bin/env python

# Peter Steiglechner, 14.10.2020

"""
Corona ABM
    - Agents have infectious states: (s, e, i_a, i_s, r, d).
            s= susceptible,
            e = exposed/latent infected
            i_a = infectious (asymptomatic)
            i_p = infectious (pre-symptomatic)
            i_s = infectious (symptomatic)
            r = recovered
            d = dead
    - The course of an infection is drawn from probability distributions according to the available data
    - Networks of interacting agents: Families, day-activity, Evening activity, Random Activity
    - Policies can be implemented: quarantine, lock-down, mask, ...
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
from create_networks import *
plt.rcParams.update({"font.size": 20})
from const import *
from additional_plot_functions import plot_network, plot_r_values
from policies import *


# Define Agent
class Agent:
    id = np.nan
    group = np.nan
    hh = -1

    state = "s"
    infectiousness = np.nan
    symptomatic = np.nan
    t_onset_symptoms = np.nan
    fatal_case = np.nan
    t_death = np.nan
    # infectious_period = [np.nan, np.nan]
    t_max_infectiousness = np.nan

    t_e = np.nan
    r = np.nan

    isolation_status = ["no isolation", np.nan]
    # [state (no isolation, social distance, quarantine), end of the isolation status]

    def catch_virus(self, t_exposure):
        """
        Determine the course of the infection, after being exposed and catching the virus
        """
        # self.state = "e"  # Exposed
        self.t_e = t_exposure

        # Probability that the infection will be symptomatic (dependent on riskgroup)
        p_s = P_SYMPTOMATIC[self.group]
        self.symptomatic = np.random.choice([True, False], p=[p_s, 1-p_s])

        if self.symptomatic:
            # Symptomatic Case
            self.state = "i_ps"

            # Incubation Time = Time from exposure to Symptom Onset
            incubation_time = INCUBATION_TIME_DIST.rvs()
            self.t_onset_symptoms = t_exposure + incubation_time
            self.t_max_infectiousness = self.t_onset_symptoms

            # Probability to die, given symptomatic case
            p_c = CASE_FATALITY_RATE[self.group]
            self.fatal_case = np.random.choice([True, False], p=[p_c, 1-p_c])

            if self.fatal_case:
                # agent dies at time t=t_death
                self.t_death = 0   # t_exposure + incubation_time + TIME_ONSET_TO_DEATH.rvs()
                # self.infectious_period = [incubation_time - START_INFECTION_BEFORE_SYMPTOM, t_death]
            else:
                # agent survives the disease, recovers (i.e. becomes non-infectious) after a certain amount of time
                pass
                # recovery_time = t_exposure + incubation_time + RECOVERY_TIME_SYMPTOMATIC.rvs()
                # self.infectious_period = [incubation_time - START_INFECTION_BEFORE_SYMPTOM, recovery_time]
        else:
            self.state = "i_a"
            self.fatal_case = False
            # Asymptomatic Case:
            # onset_asympt_infectiousness = t_exposure + (INCUBATION_TIME_DIST.rvs() - START_INFECTION_BEFORE_SYMPTOM)
            # self.t_max_infectiousness = onset_asympt_infectiousness + START_INFECTION_BEFORE_SYMPTOM
            self.t_max_infectiousness = t_exposure + INCUBATION_TIME_DIST.rvs()
            # offset_asympt_infectiousness = onset_asympt_infectiousness + RECOVERY_TIME_ASYMPTOMATIC.rvs()
            # self.infectious_period = [onset_asympt_infectiousness, offset_asympt_infectiousness]

        self.infectiousness = AGENT_INFECTIOUSNESS.rvs()  # * FACTOR_INFECTIOUSNESS

        # Alternative He2020, infectivity period starts on day 0, increases up to t_sympt,
        # self.infectious_period = [self.t_e, self.t_e+20]
        self.r = 0

        return

    # def get_infectiousness_exponential(self, t):
    #     start, end = self.infectious_period
    #     amplitude = 1 if self.symptomatic else INFECTIOUSNESS_ASYMPTOMATIC
    #     t_max = self.t_max_infectiousness
    #     if start <= t < t_max:
    #         # agent is infectious:
    #         return amplitude * self.infectiousness * I_MAX * INFECTIOUSNESS_START**((t-t_max)/(start-t_max))
    #     elif t_max <= t < end:
    #         # for symptomatic/pre-symptomatic
    #         return amplitude * self.infectiousness * I_MAX * INFECTIOUSNESS_RECOVERY**((t-t_max)/(end-t_max))
    #     else:
    #         return 0

    def get_infectiousness(self, _t):
        amplitude = 1 if self.symptomatic else INFECTIOUSNESS_ASYMPTOMATIC

        g_t = G.pdf(_t - self.t_max_infectiousness)
        relative_g_t = g_t / G.pdf(0)
        if g_t < 0.01:  # _t > self.t_max_infectiousness and
            #    self.state = "r"
            #    self.clear_infection()
            return 0
        return relative_g_t * amplitude * self.infectiousness  # * R0_CORRECTION

    def clear_infection(self):
        # Clear all
        self.symptomatic = np.nan
        self.t_onset_symptoms = np.nan
        self.fatal_case = np.nan
        # self.infectious_period = [np.nan, np.nan]
        self.infectiousness = np.nan
        self.isolation_status[0] = "no isolation"
        self.isolation_status[1] = np.nan
        return


def initialise():
    """ Initialises a list of agents """
    global agents
    # global family_net, random_net
    global N_subgroups

    # Group Index Assignment with slight random variations
    all_riskgroup_indices = np.random.choice(RISKGROUPS, p=FRACTIONS_RISKGROUPS, size=N_AGENTS)
    all_riskgroup_indices = np.sort(all_riskgroup_indices)
    N_subgroups = [np.sum(all_riskgroup_indices == _g) for _g in RISKGROUPS]

    # Group Index assignment without randomness
    '''
    N_subgroups = np.array([np.int(f * N_AGENTS) for f in FRACTIONS_RISKGROUPS])
    if not N_subgroups == np.array(FRACTIONS_RISKGROUPS) * N_AGENTS:
        print("Watch out: People in Subgroups may not add up to N_AGENTS. Choose different Fractions or N_agents ")
    all_riskgroup_indices = []
    for i, N_sub in enumerate(N_subgroups):
        all_riskgroup_indices.extend([i for _ in range(N_sub)])
    '''

    for _id in range(N_AGENTS):

        ag = Agent()

        ag.id = _id
        ag.group = all_riskgroup_indices[_id]
        # Nr of random contacts:
        # Note, these values might be overwritten (with same distribution) if this sequence is not graphical,
        # i.e. can't be translated into a network
        ag.randomContacts = stats.poisson(AVG_RANDOM_CONTACTS[ag.group]).rvs()

        ag.state = "s"
        agents.append(ag)

    symptomatic_agents = np.random.choice(agents, size=N_INFECTED_INIT)
    for ag in symptomatic_agents:
        ag.catch_virus(0)
        # ag.infectiousness = 1
    # state = "e"  # At time t=0 we start with one exposed
    return


def initialise_network(plotting=False):
    global agents, N_subgroups
    global H, F, W, R, G3

    # Create Networks
    edge_weight = 1
    R = create_Random_Network(agents, N_subgroups, AVG_RANDOM_CONTACTS, edge_weight)
    edge_weight = 2  # Just for plotting the full network in different colors
    H = create_Household_Network(agents, N_subgroups, HOUSEHOLD_FRACTIONS, edge_weight)
    edge_weight = 3  # see above
    F = create_Friends_Network(agents, N_subgroups, N_AVERAGE_FRIENDS, CORRELATIONS_FRIENDS, edge_weight)
    edge_weight = 4
    W = create_Friends_Network(agents, N_subgroups, N_AVERAGE_WORK, CORRELATIONS_WORK, edge_weight)

    g1 = nx.compose(F, H)
    g2 = nx.compose(g1, W)
    g3 = nx.compose(g2, R)
    print("Done: creating networks, now plotting them ")

    if plotting:
        plot_network(H, "Households")
        plot_network(R, "Random")
        plot_network(F, "Friends/Evening")
        plot_network(W, "Work/School")
        plot_network(g3, "Total")

    for u, v, d in G3.edges(data=True):
        d['weight'] = 1  # Set all weights of all networks to 1
    total_node_degrees = nx.adjacency_matrix(G3).sum(axis=1)
    print("An agent has on average : {:.2f}+-{:.2f} links in the net".format(total_node_degrees.mean(),
                                                                             total_node_degrees.std()))
    print("Done plotting the networks")
    return


def update(t):
    agent_list = np.random.choice(agents, size=N_AGENTS, replace=False)
    for ag in agent_list:
        if ag.state == "s":
            pass
        # if ag.state == "e":
        #    if ag.infectious_period[0] <= t: #< ag.infectious_period[1]:
        #        # Note: there's cases where the agent never becomes infected, but immediately recovers.
        #        if ag.symptomatic:
        #            ag.state = "i_ps" # pre-symptomatic
        #        else:
        #            ag.state = "i_a"

        else:
            if ag.state == "i_ps":  # any of the infectious
                if t > ag.t_onset_symptoms:
                    ag.state = "i_s"

            if ag.state[0] == "i":
                # if t >= ag.infectious_period[1] and ag.state[0] == "i":
                if t >= ag.t_max_infectiousness and ag.get_infectiousness(t) < 0.01 and not ag.fatal_case:
                    # THEN: recover from infection.
                    # end disease (fatal/recovered), clear all times.

                    if ag.fatal_case:
                        ag.state = "d"
                        ag.t_death = t
                    else:
                        ag.state = "r_s" if ag.symptomatic else "r_a"

                    ag.clear_infection()

            if ag.state[0] == "i":
                # determine neighbours
                household_contacts = list(H.adj[ag.id])
                random_contacts = list(R.adj[ag.id])
                activity_contacts = list(F.adj[ag.id]) if t == np.int(t) else []
                work_contacts = list(W.adj[ag.id]) if t == np.int(t)+0.5 else []

                all_contacts = [household_contacts, random_contacts, activity_contacts, work_contacts]

                contacts, ampl_factors = current_policy(ag, agents, t, all_contacts)

                # Meet friends/daytime, family and random
                for c, ampl_factor in zip(contacts, ampl_factors):
                    contact_person = agents[c]
                    # Potentially infect others
                    if np.random.random() < ampl_factor * ag.get_infectiousness(t):
                        if contact_person.state == "s":
                            contact_person.catch_virus(t)
                            ag.r += 1

        if ag.isolation_status[0] == "social distance" and ag.isolation_status[1] >= t:
            ag.isolation_status = ["no isolation", np.nan]


def check_policy(t):
    t_policies = np.array([pol[1] for pol in new_policies_dates])
    current_pol = new_policies_dates[int(np.where(t >= t_policies)[0][-1])][0]
    return current_pol


def run(times):
    """ Perform all update steps and return results """
    print("Do updates.")
    print("t, #s, #e, #i_a, #i_ps, #i_s, #d, #r_a, #r_s")
    global agents, N_subgroups
    global current_policy
    global R

    state_labels = ["s", "e", "i_a", "i_ps", "i_s", "d", "r_a", "r_s"]
    _results = np.empty([len(times), 9])
    for n_t, t in enumerate(times):
        update(t)
        R = create_Random_Network(agents, N_subgroups, AVG_RANDOM_CONTACTS, 1)
        # G3 = nx.compose(G2, R)

        states = [[ag.state for ag in agents].count(state) for state in state_labels]
        # non_healthy_agents = np.where(np.array([ag.state for ag in agents]) < "s")[0]
        print(t, states, end=" ")

        res = [t]
        res.extend(states)
        _results[n_t, :] = np.array(res)

        current_policy = check_policy(t)
        print(" Policy = "+current_policy.__name__, end="\n")
    return _results


if __name__ == "__main__":

    # Plot constant distributions
    plot_dists(title=None)

    """
    INIT
    """
    N_AGENTS = 2000  # Number of agents in total
    N_INFECTED_INIT = 1

    """
    Time Array
    """
    t_array = np.arange(0, 100, step=0.5)

    """
    Policies
    """
    new_policies_dates = [
        [policy_0, 0],
        [policy_1, 20],
        [policy_2, 100]
    ]
    current_policy = new_policies_dates[0][0]

    # Ensemble Run:
    all_results = []

    seeds = [1, 2, 3, 4, 5]
    for n, seed in enumerate(seeds):
        print("################# \n ###   New Run   seed {}    ### \n ################# \n".format(seed))
        np.random.seed(seed)

        agents = []  # list of all agents (ordered by index)
        N_subgroups = []
        initialise()  # Initialise all agents
        print("Done initialising all agents")

        H, F, W, R, G3 = [np.nan for _ in range(5)]
        initialise_network(plotting=False)

        # Main UPDATING
        results = run(t_array)

        # Plotting
        plot_statistics(results, title=FOLDER+"Statistics_seed{:.d}".format(seed))
        plot_r_values(n_convolve=5, title=FOLDER+"R0_seed{:.d}".format(seed))

        all_results.append(results)

    for k, s in zip(all_results, seeds):
        print("Seed: {} with results: {}".format(s, k[-1, :]))
