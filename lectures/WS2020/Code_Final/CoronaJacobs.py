#!/usr/bin/env python

# Peter Steiglechner, 14.10.2020
"""
Corona ABM
    - An agent is a person with person specific attributes and infection specific attributes
        - There are three risk groups which group-specific characteristics of the course of an infection.
        - When an agent is in contact with an infectious agent,
            she catches the virus with a probability given by the infectiousness of the infectious agent.
        - Upon catching the virus, the course (timing, type and outcome) of the infection is determined
        - These infection specific attributes are drawn from (estimated/measured) probability distributions
    - Agents are embedded in a constant Network (Watts-Strogatz-Network).
        An infected agent has contacts with all of these
    - Policies can be implemented and analysed to observe their effectiveness: e.g.
        - reducing the number of contacts in the network
        - symptomatic agents reduce contacts
        - wearing a mask (in most situations)

    The timestep of the simulation is 0.5 days
    Since the network is a small-world network, a typical Number of agents could be 1000.

"""
"""
Corona ABM: Details
    - An agent is a person with
        (1) "general attributes", defined for the whole simulation and
        (2) "infection characteristics", which are defined and determined upon catching the virus:

        general attributes
            - id (int): unique id, which will also be there node index in the network
            - group ([0,1,2]): To which risk group does the agent belong:
                - 0: low-risk (children)
                - 1: intermediate-low-risk (adults below a critical age)
                - 2: high-risk (elderly)
            - state (string) = state of infection for this agent:
                susceptible,
                exposed/latent infected
                infectious (asymptomatic)
                infectious (pre-symptomatic)
                infectious (symptomatic)
                recovered and immune from a-/symptomatic infection
                dead

    - Catch the virus
            each agent can "catch a virus". This triggers:
            function catch_virus(ag, t_exposure)
                argument: ag is the agent that catches the virus
                argument: t_exposure is the time at which the agent catches the virus.

                The function changes the state to exposed
                state = "exposed"

                The function defines/determines the infection specific attributes of the agent:
                - t_e (float): time of exposure
                - symptomatic (Bool):  Whether the infection will be symptomatic or asymptomatic.
                                        Depending on the agent's group and P_SYMPTOMATIC
                - infectious_period ([float, float]):  Time of Start and End of the infectious period.
                        the infectious period starts at earliest at t_e.
                        An incubation period is drawn from the measured distribution INCUBATION_TIME.

                        Infectiousness starts TIME_I_PRESYMPT days before the incubation time
                        Infectiousness ends TIME_I_POSTSYMPT days after the incubation time
                - t_onset_symptoms (float):
                        - For symptomatic cases: The onset of symptoms starts at t_e+incubation_period
                        - For asymptomatic cases: the incubation period has no meaning; t_onset_symptoms remains np.nan
                - fatal_case (Bool):
                    - For symptomatic cases:
                        With prob. depending on the risk-group and CFR, 
                            agent dies at the end of infectiousness.
                    All others (including asymptomatic) simply recover.
                - individual_infectiousness(float in [0,1]): 
                        Agent specific infectiousness drawn from BASE_I.
                - r (integer): Initialise the reproductive number of the agent r with 0

    - Initialisation:
        N_AGENTS agents are initialised 
            - all in the state "susceptible" (susceptible), except for N_INIT_EXPOSED agents, which are chosen randomly).
            - with probability of belonging to a certain risk group given by FRAC_RISKGROUPS

    - Initialisation of the Network:
        Any network topology can be adopted.
        The network currently used is a Watts-Strogats network.
        parameter K_WS_NETWORK determines how many links (before rewiring) emerge from each agent.
        parameter P_WS_NETWORK is the probability of each of these links to be rewired randomly
        
    - Update of all agents
        Each agent is updated one by one in an order that is redrawn anew in each timestep.
        - During an update the health state is potentially updated 
            depending on the previously determined infection course
        - If the agent is in an infectious state, it will also infect its linked contacts in the network 
            with a certain infection-state and agent-specific probability.
        
    - run and observe:
        In each timestep, an update (of all agents) is performed 
        The number of agents in each state is tracked and finally plotted.
        
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
from plot_Jacobs import *

plt.rcParams.update({"font.size": 20})  # increase the fontsize in all plots

FOLDER = "Figures/"  # Folder to save the figures in

# Risk Groups:
# Group: 0 = children (risk-free), 1 = adults, (small risk), 2 = high-risk group
RISKGROUPS = [0, 1, 2]
# Fractions of risk groups in the population: 20% children, 50% adults, 30% risk group
FRAC_RISKGROUPS = [0.2, 0.5, 0.3]  # should sum to 1

"""
Corona specific Distributions and Probabilities
"""
# BASE_I
# = Distribution from which we draw the infectiousness of an individual
#                        within the infectiousness period.
# Why this distribution?
#       Educated Guess (there are a few super spreaders, many not-so infectious people)
#       --> long tail towards high values of infectiousness
# Advanced:
#       This needs to be adjusted such that we get a mean reproductive number r (in the beginning of the
#       simulation with an entirely susceptible society) in the range of realistic values.
BASE_I = stats.beta(1, 3)

# P_SYMPTOMATIC
# = Probability of each group that, if infected with the virus how likely symptoms occur.
# Why this distribution?
#       Educated Guess (--> manifestation index)
P_SYMPTOMATIC = [0.1, 0.5, 0.8]

# INCUBATION_TIME_DIST
# = Distribution of Incubation Periods
# Why this distribution?
# estimated by
#   Lauer, et. al (2020) "The Incubation Period of Coronavirus Disease 2019 (COVID-19). From Publicly Reported
#       Confirmed Cases: Estimation and Application" (Table in Appendix 2)
#   Mean roughly 5.8 days. Long tail.
INCUBATION_PERIOD = stats.gamma(5.807, 0.948)

# CFR = CASE FATALITY RATIO = Probability that given a person has symptoms, the Covid infection leads to her death
#                       for each risk group
# Educated Guess (children "don't" die, adults seldom, risk group roughly 5%)
CFR = [0.0001, 0.005, 0.05]

# I_X
# = Infectiousness depending on the type of infection w.r.t. symptomatic infectiousness
# Why these values?
#   educated guess
#       pre-symptom cases: the transmission of the virus is smaller than with symptoms,
#       asymptomatic cases: have significantly smaller viral loads
I_SYMPT = 1  # Fix! By definition.
I_ASYMPT = 0.2       # Symptomatic cases 5 than asymptomatic
I_PRESYMPT = 0.5     # Pre-Sympt.

# TIME_I_PRE-/POST-SYMPT
# = time of potential infectiousness of an agent before/after symptom onset
# (corresponds to the same period for asymptotic cases)
#
# Why these values?
#   Some Sources
#       He, Xi, Eric H. Y. Lau, Peng Wu, Xilong Deng, Jian Wang, Xinxin Hao, Yiu Chung Lau, et al. “Temporal Dynamics
#       in Viral Shedding and Transmissibility of COVID-19.” Nature Medicine 26,
#
#       Ferguson, N., D. Laydon, G. Nedjati Gilani, N. Imai, K. Ainslie, M. Baguelin, S. Bhatia, et al. “Report 9:
#       Impact of Non-Pharmaceutical Interventions (NPIs) to Reduce COVID19 Mortality and Healthcare Demand.”
#       Report. 20, March 16, 2020.
#
#       Woelfel, Roman, Victor M. Corman, Wolfgang Guggemos, Michael Seilmaier, Sabine Zange, Marcel A. Mueller,
#       Daniela Niemeyer, et al. “Virological Assessment of Hospitalized Patients with COVID-2019.” Nature 581
#
#       ECDC (2020): https://www.ecdc.europa.eu/en/covid-19/facts/questions-answers-basic-facts
#
#   - He (2020): roughly 2 days before symptom onset;
#   - Ferguson (2020) uses 0.5 days before
#   Note: He(2020) actually corrected recently to ca. 5 days before symptom onset
#   - Woelfel (2020) says "virus shedding was very high during the first week of symptoms", peak at day 4.
#   - He (2020) (corrected): The infeciousness peaks around symptom onset.
#   - ECDC(2020) says: infectiousness starts up to 2 to 2.5 days before, with a peak at symptom onset, then fast decline
#
TIME_I_PRESYMPT = 2  # days before symptom onset
TIME_I_POSTSYMPT = 4  # days after symptom onset


class Agent:
    """ Empty class, filled during initialise() and potentially during catch_virus(agent, t) """
    r = np.nan  # Needed for the analysis of the reproductive number. np.nan means this person will not be counted
    # in averaging the number of people each agent will affect.
    t_e = np.nan  # Same as above
    pass

def initialise():
    """
    Initialise all agents.
    :return:
    """
    global agents
    agents = []

    for id in range(N_AGENTS):
        ag = Agent()
        ag.id = id					# Unique ID
        ag.state = "susceptible"	# health state
        ag.group = np.random.choice(RISKGROUPS, p=FRAC_RISKGROUPS)      # Age-/Rsik Group
        agents.append(ag)

    # Let N_INFECTED_INIT randomly chosen agents catch the virus at t=0
    infected_agents = np.random.choice(agents, size=N_INFECTED_INIT)
    for ag in infected_agents:
        catch_virus(ag, 0)		# agent ag catches virus at time 0.

    return


def catch_virus(ag, t_exposure):
    """
    Determine the course of the infection, after being exposed and catching the virus.
    An agent ag has been in contact (i.e. exposed) with an infectious individual AND has been caught the virus.
    - agent's state switches to exposed (and t_e to the exposure time).
    - agent's infection specific properties are determined:
        - symptomatic
        - infectious_period
        - (t_onset_symptoms)
        - (fatal_case)
        - infectiousness
    - the agent's r value (how many agents did this person infect) is defined and initialised with 0.

    :param ag: (object). Instance of class Agent. The agent which was exposed to the virus and caught it.
    :param t_exposure: (float). The time at which the agent catches the virus
    :return: --   (just changing agent ag's attributes)
    """

    ag.state = "exposed"  # Exposed
    ag.t_e = t_exposure

    # Probability that the infection will be symptomatic (dependent on riskgroup)
    p_s = P_SYMPTOMATIC[ag.group]
    ag.symptomatic = np.random.choice([True, False], p=[p_s, 1 - p_s])

    incubation_period = INCUBATION_PERIOD.rvs()
    ag.infectious_period = [
        ag.t_e + incubation_period - TIME_I_PRESYMPT,
        ag.t_e + incubation_period + TIME_I_POSTSYMPT
    ]
    if ag.symptomatic:
        # Symptomatic Case
        ag.t_onset_symptoms = ag.t_e + incubation_period

        # Probability to die, given symptomatic case
        p_d = CFR[ag.group]
        ag.fatal_case = np.random.choice([True, False], p=[p_d, 1 - p_d])
    else:
        ag.t_onset_symptoms = np.nan
        ag.fatal_case = False
    ag.base_infectiousness = BASE_I.rvs()   # * FACTOR_INFECTIOUSNESS

    ag.r = 0
    return


def initialise_network(_seed, plotting=True, **kwargs_network):
    """
    Create Network (nodes taken from agents, links created randomly) and
    :param plotting: (Bool, default = True)
    :param kwargs_network: dict.  for Watts-Strogatz Network: We need to give k and p. Hence, "k=10, p=2"
    :return ws_net: The network including all links between the nodes (agents).
    """
    n_ws = len(agents)  # All agents are nodes, their id is equal to their node index
    k_ws = kwargs_network["k"]  # with how many direct neighbours is an agent connected
    p_ws = kwargs_network["p"]  # probability for a link to flip
    ws_net = nx.watts_strogatz_graph(n_ws, k_ws, p_ws, seed=_seed)
    if plotting:
        print("Plotting Network...", end="")
        plot_net(ws_net, agents)
        print("DONE.")
    return ws_net


def update(t_now):
    """
    Perform one update step, i.e. update all agents in random order!
    :param t_now:
    :return:
    """
    queue = np.random.choice(agents, size=N_AGENTS, replace=False)
    # CHECK AGAIN: SHOULD I USE SYNCHRONOUS ORDER OR ASYNCHRONOUS.
    for ag in queue:

        if ag.state == "susceptible":
            # susceptible: Do nothing and "wait to be infected"
            pass

        if ag.state == "exposed":
            # exposed and caught the virus: Check if time for infectiousness start is exceeded
            if t_now >= ag.infectious_period[0]:
                if ag.symptomatic:
                    ag.state = "inf_presympt"  # pre-symptomatic
                else:
                    ag.state = "inf_asympt"  # asymptomatic

        if ag.state == "inf_presympt":
            # infectious: Check if time for symptom onset is exceeded
            if t_now >= ag.t_onset_symptoms:
                ag.state = "inf_sympt"

        if ag.state[0:3] == "inf":  # any of the infectious states
            if t_now >= ag.infectious_period[1]:
                # Recover from infection.
                # end disease (fatal/recovered), clear all times.
                if ag.fatal_case:
                    ag.state = "dead"  # dead
                else:
                    ag.state = "recovered"  # r
        '''
        INFECT OTHERS
        '''
        if ag.state[0:3] == "inf":
            # Get Infectiousness:
            if ag.state == "inf_sympt":
                p_i_now = ag.base_infectiousness * I_SYMPT
            elif ag.state == "inf_presympt":
                p_i_now = ag.base_infectiousness * I_PRESYMPT
            elif ag.state == "inf_asympt":
                p_i_now = ag.base_infectiousness * I_ASYMPT

            # potentially infect others!
            # determine neighbours
            linked_contacts = list(Net.adj[ag.id])

            for c in linked_contacts:
                contact_person = agents[c]
                # Potentially infect this contact person.
                if np.random.random() < p_i_now:
                    if contact_person.state == "susceptible":
                        catch_virus(contact_person, t_now)
                        ag.r += 1


def run(_seed):
    """ Run the simulation with the current random seed """
    global agents, Net
    np.random.seed(_seed)

    """
    INITIALISE
    """
    agents = []  # list of all agents (ordered by index)
    initialise()  # Initialise all agents
    print("Done initialising all agents")
    Net = initialise_network(_seed, plotting=False, k=K_WS_NETWORK, p=P_WS_NETWORK)

    """
    MAIN UPDATING
    """
    print("Do updates.")
    states = ["susceptible", "exposed", "inf_asympt", "inf_presympt", "inf_sympt", "recovered", "dead"]
    print("t, #" + ", #".join(states))
    results = np.empty([len(T_ARRAY), len(states)])

    for n, _t in enumerate(T_ARRAY):
        # update one time-step
        update(_t)
        # Store Results
        states_of_agents = [ag.state for ag in agents]
        N_s = states_of_agents.count("susceptible")
        N_e = states_of_agents.count("exposed")
        N_ia = states_of_agents.count("inf_asympt")
        N_ips = states_of_agents.count("inf_presympt")
        N_is = states_of_agents.count("inf_sympt")
        N_r = states_of_agents.count("recovered")
        N_d = states_of_agents.count("dead")

        results[n, :] = np.array([N_s, N_e, N_ia, N_ips, N_is, N_r, N_d])

        print(_t, N_s, N_e, N_ia, N_ips, N_is, N_r, N_d)

    '''
    Plotting
    '''
    title = r"$N_{Agents}$" + r"={:d}, $N_e(t=0)$={:d}, Net: Watts-Strogatz ({:d}, {:.2f}) seed={:d}".format(
        N_AGENTS, N_INFECTED_INIT, K_WS_NETWORK, P_WS_NETWORK, _seed)
    filename = "AggregateResults_N-{:d}-{:d}_WS({:d},{:.0e})_seed{:d}".format(
        N_AGENTS, N_INFECTED_INIT, K_WS_NETWORK, P_WS_NETWORK, _seed)
    plot_statistics(T_ARRAY, results, states, title=title, filename=FOLDER + filename)

    return results


if __name__ == "__main__":
    """ 
    MAIN
    """
    N_AGENTS = 1000  # Number of agents in total
    N_INFECTED_INIT = 2  # Number of agents in state "exposed" at t=0.

    """
    Time Array
    """
    T_ARRAY = np.arange(0, 200, step=0.5)

    """
    The Network
    """
    K_WS_NETWORK = 4
    P_WS_NETWORK = 0.2

    '''
    RUN
    '''
    seed = 10     # Set random seed, so that running the model again with the same seed results in the same output.
    agents = []     # Will be loaded with a list of agents
    Net = {}        # Will be loaded with the network, i.e. with Nodes and Links.
    results = run(seed)



    """
    ADVANCED 2:
    Plot R0
    """
    plot_r_values(agents, fname=FOLDER+"R0values_seed"+str(seed))




    """ 
    ADVANCED:     
    #   - CHECK DIFFERENT k OF WATTS_STORGARTS MODEL 
    #   - PERFORM STATISTICAL RUNS / ENSMEBLE RUNS
    #   - AND PLOT 
    #       (1) HOW OFTEN OUTBREAKS OCCUR, 
    #       (2) THE FRACTION OF DEATHS/INFECTED IN SUCH OUTBREAKS
    #       OVER THE USED k VALUES
    # 
    """
    """
    INSTRUCTIONS:
    Replace the code in the if__name__=="__main__": by the following
    """
    # final_fractions_networks = []
    # k_s = [2, 4, 6, 8, 10]
    # P_WS_NETWORK = 0.2
    # for k in k_s:
    #     ensemble_results = []
    #     seeds = np.arange(0, 10)
    #     for seed in seeds:
    #         results = run(seed)
    #         ensemble_results.append(results[-1, 1:])
    #
    #     ensemble_results = np.array(ensemble_results)
    #     print(ensemble_results)
    #     outbreaks = np.where(ensemble_results[:, 0] < 0.5 * N_AGENTS)[0]
    #     fraction_outbreaks = len(outbreaks) / len(seeds)
    #     if len(outbreaks) > 0:
    #         fraction_infected_in_outbreak = 1 - np.mean([ensemble_results[o, 0] / N_AGENTS for o in outbreaks])
    #         fraction_dead_in_outbreak = np.mean([ensemble_results[o, -1] / N_AGENTS for o in outbreaks])
    #     else:
    #         fraction_infected_in_outbreak = 0
    #         fraction_dead_in_outbreak = 0
    #
    #     final_fractions_networks.append([fraction_outbreaks, fraction_infected_in_outbreak, fraction_dead_in_outbreak])
    #
    # final_fractions_networks = np.array(final_fractions_networks)
    # fractions_outbreaks = final_fractions_networks[:, 0]
    # fractions_infected_in_outbreak = final_fractions_networks[:, 1]
    # fractions_dead_in_outbreak = final_fractions_networks[:, 2]
    #
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot(111)
    # ax.plot(k_s, fractions_outbreaks, lw=3, label="Fraction Outbreaks", color="b")
    # ax.legend(loc="upper left")
    # ax2 = ax  # .twinx()
    # ax2.plot(k_s, fractions_infected_in_outbreak, lw=3, label=r"Fraction Infected in Outbreak $P(e|outbreak)$",
    #          color="g")
    # ax2.legend(loc="lower left")
    # ax3 = ax.twinx()
    # ax3.plot(k_s, fractions_dead_in_outbreak, lw=3, label=r"Fraction Dead in Outbreak $P(d|outbreak)$", color="recovered")
    # ax3.legend(loc="center right")
    # fig.tight_layout()
    # plt.savefig(
    #     FOLDER + "Fractions_WS(p={:.0e})_varied-k_N-{:d}-{:d}.pdf".format(P_WS_NETWORK, N_AGENTS, N_INFECTED_INIT),
    #     bbox_inches="tight")
    # plt.show()
