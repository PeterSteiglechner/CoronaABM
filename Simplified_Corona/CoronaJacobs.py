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
Corona ABM
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
                s = susceptible,
                e = exposed/latent infected
                i_a = infectious (asymptomatic)
                i_p = infectious (pre-symptomatic)
                i_s = infectious (symptomatic)
                r_a = recovered and immune from asymptomatic infection
                r_s = recovered and immune from symptomatic infection
                d = dead

        catching the virus
            each agent can "catch a virus". This triggers:
            function catch_virus(self, t_exposure)
                argument: t_exposure is the time at which the agent catches the virus.

                The function changes the state to exposed
                state = "e"

                The function defines/determines the infection specific attributes:
                - t_e (float): time of exposure
                - symptomatic (Bool):  Whether the infection will be symptomatic or asymptomatic.
                                        Depending on the agent's group and P_SYMPTOMATIC
                - infectious_period ([float, float]):  Time of Start and End of the infectious period.
                        the infectious period starts at earliest at t_e.
                        An incubation period is drawn from the measured distribution INCUBATION_TIME_DIST.

                        Infectiousness starts TIME_PRESYMPTOM_INFECTION days before the incubation time
                        Infectiousness ends TIME_PRESYMPTOM_INFECTION days after the incubation time
                - t_onset_symptoms (float):
                        - For symptomatic cases: The onset of symptoms starts at t_e+incubation_period
                        - For asymptomatic cases: the incubation period has no meaning; t_onset_symptoms remains np.nan
                - fatal_case (Bool):
                    - For symptomatic cases:
                        With prob. depending on the risk-group and CASE_FATALITY_RATE, 
                            agent dies at the end of infectiousness.
                    All others (including asymptomatic) simply recover.
                - individual_infectiousness(float in [0,1]): 
                        Agent specific infectiousness drawn from AGENT_INFECTIOUSNESS.

    - Initialisation:
        N_AGENTS agents are initialised 
            - all in the state "s" (susceptible), except for N_INIT_EXPOSED agents, which are chosen randomly).
            - with probability of belonging to a certain risk group given by FRACTIONS_RISKGROUPS


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
RISKGROUPS = [0, 1, 2]  # Group: 0 = children (risk-free), 1 = adults, (small risk), 2 = high-risk group.
FRACTIONS_RISKGROUPS = [0.2, 0.5, 0.3]  # Fractions of risk groups

"""
Corona specific Distributions and Probabilities
"""
# AGENT_INFECTIOUSNESS = Distribution from which we draw the infectiousness of an symptomatic individual
#                           within the infectiousness period.
# Educated Guess (there are a few super spreaders, many not-so infectious people) --> long tail to high values
# this needs to be adjusted such that we get a mean reproductive number r (in the beginning of the simulation with an
# entirely susceptible society) in the range of realistic values.
AGENT_INFECTIOUSNESS = stats.beta(1, 3)

# P_SYMPTOMATIC = Probability of each group that, if infected with the virus how likely symptoms occur.
# Educated Guess (--> manifestation index)
P_SYMPTOMATIC = [0.1, 0.5, 0.8]  # should sum to 1

# INCUBATION_TIME_DIST = Distribution of Incubation Periods
# estimated by
# Lauer, et. al (2020) "The Incubation Period of Coronavirus Disease 2019 (COVID-19). From Publicly Reported
# Confirmed Cases: Estimation and Application" (Table in Appendix 2)
# Mean roughly 5.8 days. Long tail.
INCUBATION_PERIOD_DIST = stats.gamma(5.807, 0.948)
# Γ(α,λ)=Γ(2.810,0.419) # https://www.medrxiv.org/content/10.1101/2020.10.20.20216143v1


# CASE_FATALITY_RATE = Probability that given a person has symptoms, the Covid infection leads to her death
#                       for each risk group
# Educated Guess (children "don't" die, adults seldom, risk group roughly 5%)
CASE_FATALITY_RATE = [0.0001, 0.005, 0.05]

# INFECTIOUSNESS_ = Infectiousness depending on the type of infection w.r.t. symptomatic infectiousness
# educated guess (without symptoms, the distribution of virus is smaller than with symptoms,
#   asymptomatic cases have significantly smaller viral loads)
INFECTIOUSNESS_SYMPTOMATIC = 1  # Fix!
INFECTIOUSNESS_ASYMPTOMATIC = 0.2
INFECTIOUSNESS_PRESYMPTOMATIC = 0.5

# TIME_PRE-/POST-SYMPTOM_INFECTION = time of potential infectiousness of an agent before/after symptom onset
#       (corresponds to the same period for asymptotic cases)
# Educated guess:
#       - He (2020), says roughly 2 days before symptom onset; Ferguson (2020) uses 0.5 days before
# Note: He(2020) actually corrected recently to ca. 5 days before symptom onset
#       - ... (2020) says viral load after 7 days after symptom onset very low;
#         ... (2020) uses exponentially declining viral load after symptom onset

# https://www.ecdc.europa.eu/en/covid-19/facts/questions-answers-basic-facts
# says: up to 2 to 2.5 days before, and peak at symptom onset. then fast decline

TIME_PRESYMPTOM_INFECTION = 2
TIME_POSTSYMPTOM_INFECTION = 4


class Agent:
    """ define an agent and its properties """
    def __init__(self, _id):
        """ initialise Agent """
        # General properties
        self.id = _id
        self.state = "s"  # susceptible
        self.group = np.random.choice(RISKGROUPS, p=FRACTIONS_RISKGROUPS)

        # infection specific attributes (for later)
        # np.nan --> they are not yet defined.
        self.t_e = np.nan  # time of catching the virus
        self.symptomatic = np.nan
        self.t_onset_symptoms = np.nan
        self.fatal_case = np.nan
        self.infectious_period = [np.nan, np.nan]
        self.base_infectiousness = np.nan

        # For analysis of reproduction number r over time.
        self.r = np.nan
        return

    def catch_virus(self, t_exposure):
        """
        Determine the course of the infection, after being exposed and catching the virus.
        An agent has been in contact with an infectious individual AND has been exposed (and caught) the virus.
        - agent's state switches to e (and t_e to the exposure time).
        - agent's infection specific properties are determined:
            - symptomatic
            - infectious_period
            - (t_onset_symptoms)
            - (fatal_case)
            - infectiousness
        - the agent's r value (how many agents did this person infect) is defined and initialised with 0.

        :param t_exposure: (float). The time at which the agent catches the virus
        :return: --   (just changing agent's attributes)
        """

        self.state = "e"  # Exposed
        self.t_e = t_exposure

        # Probability that the infection will be symptomatic (dependent on riskgroup)
        p_s = P_SYMPTOMATIC[self.group]
        self.symptomatic = np.random.choice([True, False], p=[p_s, 1 - p_s])

        incubation_period = INCUBATION_PERIOD_DIST.rvs()
        self.infectious_period = [
            self.t_e + incubation_period - TIME_PRESYMPTOM_INFECTION,
            self.t_e + incubation_period + TIME_POSTSYMPTOM_INFECTION
        ]
        if self.symptomatic:
            # Symptomatic Case
            self.t_onset_symptoms = self.t_e + incubation_period

            # Probability to die, given symptomatic case
            p_c = CASE_FATALITY_RATE[self.group]
            self.fatal_case = np.random.choice([True, False], p=[p_c, 1 - p_c])
        else:
            self.t_onset_symptoms = np.nan
            self.fatal_case = False
        self.base_infectiousness = min(1, AGENT_INFECTIOUSNESS.rvs())  # * FACTOR_INFECTIOUSNESS

        self.r = 0
        return

    def get_infectiousness_simplified(self, t_now):
        """
        Returns the infectiousness of an agent
        in her current state (at time t) of infection (a-, pre-, or symptomatic)
        and given her agent-specific infectiousness

        :param t_now: (float)
        :return current_infectiousness: (float) probability of infecting someone at time t
        """
        start, end = self.infectious_period  # just for security. This is not necessary.
        if self.symptomatic:
            # Agent's infection is pre- or symptomatic
            if start <= t_now < self.t_onset_symptoms:
                current_infectiousness = min(1., self.base_infectiousness * INFECTIOUSNESS_PRESYMPTOMATIC)
            else:
                current_infectiousness = min(1., self.base_infectiousness * INFECTIOUSNESS_SYMPTOMATIC)
        else:
            # Agent's infection is asymptomatic
            current_infectiousness = min(1., self.base_infectiousness * INFECTIOUSNESS_ASYMPTOMATIC)
        return current_infectiousness


def initialise():
    """
    Initialises a list of agents (with a few people catching the virus at t=0)
    :param: --
    :return: --
    """
    global agents

    for _id in range(N_AGENTS):
        # Define new agent
        ag = Agent(_id)
        agents.append(ag)

    # [N_0, N_1, N_2] how many agents in each risk group.
    # N_subgroups = [np.sum([ag.group == g for ag in agents]) for g in RISKGROUPS]

    # Let N_INFECTED_INIT randomly chosen agents catch the virus at t=0
    symptomatic_agents = np.random.choice(agents, size=N_INFECTED_INIT)
    for ag in symptomatic_agents:
        ag.catch_virus(0)
    return


def initialise_network(agent_list, plotting=True, **kwargs_network):
    """
    Create Network (nodes taken from agents, links created randomly) and
    :param agent_list: list(objects of Agent())
    :param plotting: (Bool, default = True)
    :param kwargs_network: dict.  for Watts-Strogatz Network: We need to give k and p. Hence, "k=10, p=2"
    :return ws_net: The network including all links between the nodes (agents).
    """
    n_ws = len(agents)  # All agents are nodes, their id is equal to their node index
    k_ws = kwargs_network["k"]  # with how many direct neighbours is an agent connected
    p_ws = kwargs_network["p"]  # probability for a link to flip
    ws_net = nx.watts_strogatz_graph(n_ws, k_ws, p_ws)
    if plotting:
        print("Plotting Network...", end="")
        plot_net(ws_net, agent_list)
        print("DONE.")
    return ws_net


def update(t_now):
    """
    Perform one update step, i.e. update all agents in random order!
    :param t_now:
    :return:
    """
    agent_list = np.random.choice(agents, size=N_AGENTS, replace=False)
    # CHECK AGAIN: SHOULD I USE SYNCHRONOUS ORDER OR ASYNCHRONOUS.
    for ag in agent_list:
        if ag.state == "s":
            pass
        if ag.state == "e":
            if ag.infectious_period[0] <= t_now:
                if ag.symptomatic:
                    ag.state = "i_ps"  # pre-symptomatic
                else:
                    ag.state = "i_a"

        if ag.state == "i_ps":  # any of the infectious
            if t_now > ag.t_onset_symptoms:
                ag.state = "i_s"

        if t_now >= ag.infectious_period[1] and (ag.state[0] == "i" or ag.state == "e"):
            # THEN: recover from infection.
            # end disease (fatal/recovered), clear all times.
            # Note: there's cases where the agent never becomes infected
            if ag.fatal_case:
                ag.state = "d"
            else:
                ag.state = "r_s" if ag.symptomatic else "r_a"

            # Clear all
            ag.symptomatic = np.nan
            ag.t_onset_symptoms = np.nan
            ag.fatal_case = np.nan
            ag.infectious_period = [np.nan, np.nan]
            ag.infectiousness = np.nan

        '''
        INFECT OTHERS
        '''
        if ag.state[0] == "i":
            # determine neighbours
            linked_contacts = list(Net.adj[ag.id])
            # WEIGHTS? MIGHT BE A NICE IDEA TO PLAY AROUND?
            # HOW MANY PEOPLE NEED TO SOCIALLY DISTANCE TO FLATTEN THE CURVE.

            for c in linked_contacts:
                contact_person = agents[c]
                # Potentially infect others
                if np.random.random() < ag.get_infectiousness_simplified(t_now):
                    if contact_person.state == "s":
                        contact_person.catch_virus(t_now)
                        ag.r += 1


def run(seed):
    """ Run the simulation """
    global agents, Net
    np.random.seed(seed)

    """
    INITIALISE
    """
    agents = []  # list of all agents (ordered by index)
    initialise()  # Initialise all agents
    print("Done initialising all agents")
    Net = initialise_network(agents, plotting=False, k=K_WS_NETWORK, p=P_WS_NETWORK)

    """
    MAIN UPDATING
    """
    print("Do updates.")
    states = ["s", "e", "i_a", "i_ps", "i_s", "r_a", "r_s", "d"]
    print("t, #" + ", #".join(states))
    results = np.empty([len(T_ARRAY), len(states) + 1])

    for n, t in enumerate(T_ARRAY):
        # update one time-step
        update(t)
        # Store Results
        agent_states = [[ag.state for ag in agents].count(state) for state in states]
        print(t, agent_states)
        res = [t]
        res.extend(agent_states)
        results[n, :] = np.array(res)

    '''
    Plotting
    '''
    title = r"$N_{Agents}$" + r"={:d}, $N_e(t=0)$={:d}, Net: Watts-Strogatz ({:d}, {:.2f}) seed={:d}".format(
        N_AGENTS, N_INFECTED_INIT, K_WS_NETWORK, P_WS_NETWORK, seed)
    filename = "AggregateResults_N-{:d}-{:d}_WS({:d},{:.0e})_seed{:d}".format(
        N_AGENTS, N_INFECTED_INIT, K_WS_NETWORK, P_WS_NETWORK, seed)
    plot_statistics(results, states, title=title, filename=FOLDER + filename)

    return results


if __name__ == "__main__":
    """ 
    MAIN
    """
    N_AGENTS = 1000  # Number of agents in total
    N_INFECTED_INIT = 2  # Number of agents in state "e" at t=0.

    """
    Time Array
    """
    T_ARRAY = np.arange(0, 200, step=0.5)


    """
    The Network
    """
    K_WS_NETWORK = 6
    P_WS_NETWORK = 0.2


    seed = 1
    agents = []     # Will be loaded with a list of agents
    Net = {}        # Will be loaded with the network, i.e. with Nodes and Links.
    results = run(seed)







    """# ADVANCED:     
    # CHECK DIFFERENT k OF WATTS_STORGARTS MODEL 
    # PERFORM STATISTICAL RUNS / ENSMEBLE RUNS
    # AND PLOT 
    #       (1) HOW OFTEN OUTBREAKS OCCUR, 
    #       (2) THE FRACTION OF DEATHS/INFECTED IN SUCH OUTBREAKS
    # over the used k.
    # 
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
    # ax3.plot(k_s, fractions_dead_in_outbreak, lw=3, label=r"Fraction Dead in Outbreak $P(d|outbreak)$", color="r")
    # ax3.legend(loc="center right")
    # fig.tight_layout()
    # plt.savefig(
    #     FOLDER + "Fractions_WS(p={:.0e})_varied-k_N-{:d}-{:d}.pdf".format(P_WS_NETWORK, N_AGENTS, N_INFECTED_INIT),
    #     bbox_inches="tight")
    # plt.show()
