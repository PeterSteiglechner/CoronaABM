# %% [markdown]
# # Lecture 11: Modeling the spread of Covid-19
# 
# by Peter Steiglechner, peter.steiglechner@gmail.com, 
# 
# - developed from March 2020 to Oct 2020
# - last updated Nov 2022
# 
# 

# %% [markdown]
# ### Overview 
# - Each agent is a person
# - Agents are categorised into three age groups 
# - Infection dynamics:
#     - A susceptible agent may catch the virus when in contact with an infectious agent. The probability to catch the virus is given by the infectiousness of the infectious agent.
#     - Upon becoming infected, the infection course (timing, type, and outcome) is determined stochastically based on distributions. These distributions are based on educated guesses or inferred from data. Some characteristics also depend on the age of the infected agent.
# - Agents are embedded in a fixed interaction network (here a Watts-Strogatz-Network). Links between agents (nodes) represent physical contact that may result in an spreading event from one agent to the other. All agents interact asynchronously with one of their neighbours in one time step. 
# - Policies can be implemented and analysed to investigate their effectiveness: e.g.
#     - reducing the number of contacts in the network
#     - reduce contacts for symptomatic agents
#     - reduce infectiousness by wearing a mask
# 
# The timestep of the simulation is 0.5 day.
# 
# The number of agents is set to 1000 including children, adults (with low risk) and elderly (with higher risk).

# %% [markdown]
# ### Documentation 
# For details, refer to the Lecture slides 11 of "Modelling and Analysis of Complex Systems" (Agostino Merico) Jacobs University 2022

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
from plot_Jacobs import *

plt.rcParams.update({"font.size": 20})  # increase the fontsize in all plots




# %% [markdown]
# # Global fixed parameters

# %%
AGE_GROUPS = [ "child" , "adult" , "elderly"]
# Fractions of age groups in the population: 20% children, 50% adults, 30% elderly
FRACTION_AGE_GROUPS = [0.2, 0.5, 0.3]  # should sum to 1

"""
Corona specific distributions and probabilities
"""
# BASE_I
# = Distribution from which we draw the infectiousness of an individual
# Why this distribution?
#       Educated Guess (there are a few super spreaders, many not-so infectious people)
#       --> long tail towards high values of infectiousness
# For advanced assignment:
#       This needs to be adjusted such that we get a mean reproductive number R (in the beginning of the
#       simulation where (nearly) all agents are susceptible) close to realistic values.
BASE_I = stats.beta(1, 3)

# P_SYMPTOMATIC
# = Probability of a member of a specific age group to have a symptomatic infection.
# Why this distribution?
#       Educated Guess (related to the --> manifestation index)
P_SYMPTOMATIC = {"child": 0.1, "adult": 0.5, "elderly": 0.8}

# INCUBATION_TIME_DIST
# = Distribution of incubation periods
# Why this distribution?
# estimated by
#   Lauer, et. al (2020) "The Incubation Period of Coronavirus Disease 2019 (COVID-19). From Publicly Reported
#       Confirmed Cases: Estimation and Application" (Table in Appendix 2)
#   Mean roughly 5.8 days. Long tail.
INCUBATION_PERIOD = stats.gamma(5.807, 0.948)

# CFR = CASE FATALITY RATIO = Probability that given a person has symptoms, the Covid infection leads to her death
#                       for each risk group
# Educated Guess (children "don't" die, adults seldom, risk group roughly 5%)
CFR = {"child": 0.0001, "adult": 0.005, "elderly": 0.05}

# I_X
# = relative infectiousness depending on the type of infection w.r.t. symptomatic infectiousness
# Why these values?
#   educated guess
#       pre-symptom cases: the transmission of the virus is smaller than with symptoms,
#       asymptomatic cases: have significantly smaller viral loads
RELATIVE_INFECTIOUSNESS = {"infectious_presymptomatic": 0.5, "infectious_symptomatic": 1, "infectious_asymptomatic": 0.2}

# TIME_I_PRE-/POST-SYMPT
# = time of potential infectiousness of an agent before/after symptom onset
# (corresponds to the same period for asymptotic cases)
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



# %% [markdown]
# # Simulation specific parameter choices

# %%
FOLDER = "figures/"  # Folder to save the figures in

N_AGENTS = 1000  # Number of agents in total
N_INFECTED_INIT = 2  # Number of agents in state "exposed" at t=0.

"""
Time Array
"""
T_ARRAY = np.arange(0, 400, step=1)



# %% [markdown]
# # Agent class

# %%
class Agent:
    """ Empty class, filled during initialise() and potentially during catch_virus(agent, t) """
    pass

# %% [markdown]
# # Initialise

# %%
def initialise():
    """
    Initialise all agents.
    :return:
    """
    global agents
    agents = []

    for i in range(N_AGENTS):
        ag = Agent()
        ag.id = i					# Unique ID
        ag.health_state = "susceptible"	# health state
        ag.age = np.random.choice(AGE_GROUPS, p=FRACTION_AGE_GROUPS)
        ag.r = np.nan  # LATER: for calculting the r-value
        ag.t_e = np.nan  # LATER: for plotting infection dynamics.
        agents.append(ag)

    # Let N_INFECTED_INIT randomly chosen agents catch the virus at t=0
    infected_agents = np.random.choice(agents, size=N_INFECTED_INIT)
    for ag in infected_agents:
        catch_virus(ag, 0)		# agent ag catches virus at time 0.

    return


# %%
def initialise_network(_seed, plotting=True, **kwargs_network):
    """
    Create Network (nodes taken from agents, links created randomly) and
    :param plotting: (Bool, default = True)
    :param kwargs_network: dict.  for Watts-Strogatz Network: We need to give k and p. Hence, "k=10, p=2"
    :return ws_net: The network including all links between the nodes (agents).
    """
    n_ws = len(agents)  # All agents are nodes, their id is equal to their node index
    k_ws = kwargs_network["k"]  # number of links to left/right neighbours per agent
    p_ws = kwargs_network["p"]  # probability for a link to flip
    ws_network = nx.watts_strogatz_graph(n_ws, k_ws, p_ws, seed=int(_seed))
    
    if plotting:
        print("Plotting Network...", end="")
        plot_net(ws_network, agents)
        print("DONE.")
    
    return ws_network

# %% [markdown]
# # Update procedure and helper functions

# %%

def update(t_now):
    """
    Perform one update step, i.e. update all agents in random order!
    :param t_now:
    :return:
    """
    queue = np.random.choice(agents, size=N_AGENTS, replace=False)
    for ag in queue:

        if ag.health_state == "susceptible":
            # agent is susceptible: Do nothing and "wait to be infected"
            pass

        if ag.health_state == "exposed":
            # agent is exposed and caught the virus: Check if time for infectiousness start is exceeded
            if t_now >= ag.infectious_period[0]:
                if ag.symptomatic:
                    ag.health_state = "infectious_presymptomatic"
                else:
                    ag.health_state = "infectious_asymptomatic" 

        if ag.health_state == "infectious_presymptomatic":
            # infectious: Check if time for symptom onset is exceeded
            if t_now >= ag.t_onset_symptoms:
                ag.health_state = "infectious_symptomatic"

        if "infectious" in ag.health_state: 
            if t_now >= ag.infectious_period[1]:
                # Recover or die from infection.
                if ag.fatal_case:
                    ag.health_state = "dead" 
                else:
                    ag.health_state = "recovered"  
            
            # potentially infect others
            infect_others(ag, t_now)

# %%
def catch_virus(ag, t_exposure):
    """
    Determine the course of the infection upon being exposed and catching the virus.
    An agent ag has been in contact (i.e. exposed) with an infectious individual AND has caught the virus.
    - the agent's state switches to exposed (and t_e to the exposure time).
    - we determine the agent's infection specific properties:
        - symptomatic?
        - infectious_period
        - t_onset_symptoms (if symptomatic)
        - fatal? (if symptomatic)
        - individual_infectiousness
    - we define the agent's r value (how many agents did this person infect) and initialise it with 0.

    :param ag: (object). Instance of class Agent. The agent that was exposed to the virus and caught it.
    :param t_exposure: (float). The time at which the agent catches the virus
    :return: --   (just changing agent ag's attributes)
    """

    ag.health_state = "exposed"  # Exposed
    ag.t_e = t_exposure

    # Probability that the infection will be symptomatic (dependent on risk_group)
    p_s = P_SYMPTOMATIC[ag.age]
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
        p_d = CFR[ag.age]
        ag.fatal_case = np.random.choice([True, False], p=[p_d, 1 - p_d])
    else:
        ag.t_onset_symptoms = np.nan
        ag.fatal_case = False
    ag.base_infectiousness = BASE_I.rvs()   # * FACTOR_INFECTIOUSNESS

    ag.r = 0
    return


                        
    

# %%
def infect_others(ag, t_now):
    """
    An agent ag can infect its linked contacts each with a probability dependent on its base infectiousness and state. 

    :param ag: (object). instance of class Agent.
    :param t_now: (float). current time
    :return: --  
    """
    global agents, network
    if "infectious" in ag.health_state:
            # Get Infectiousness:
            p_i = ag.base_infectiousness * RELATIVE_INFECTIOUSNESS[ag.health_state]

            # potentially infect others!
            # determine neighbours
            linked_contacts = list(network.adj[ag.id])  # very slow!

            for c in linked_contacts:
                contact_person = agents[c]
                # Potentially infect this contact person.
                if contact_person.health_state == "susceptible":
                    if np.random.random() < p_i:
                        catch_virus(contact_person, t_now)
                        ag.r += 1
    return 

# %% [markdown]
# ## Observe

# %%
def observe(agents):
    states_of_agents = [ag.health_state for ag in agents]
    N_s = states_of_agents.count("susceptible")
    N_e = states_of_agents.count("exposed")
    N_ia = states_of_agents.count("infectious_asymptomatic")
    N_ips = states_of_agents.count("infectious_presymptomatic")
    N_is = states_of_agents.count("infectious_symptomatic")
    N_r = states_of_agents.count("recovered")
    N_d = states_of_agents.count("dead")
    return np.array([N_s, N_e, N_ia, N_ips, N_is, N_r, N_d])

# %% [markdown]
# ## Run function: 
# 1) initialise agents and network
# 2) simulate
# 3) observe
# 4) plot observations

# %%
def run(_seed, k_ws, p_ws):
    """ Run the simulation with the current random seed """
    global agents, network
    np.random.seed(_seed)

    """
    INITIALISE
    """
    agents = []  # list of all agents (ordered by index)
    initialise()  # Initialise all agents
    network = initialise_network(_seed, plotting=False, n=N_AGENTS, k=k_ws, p=p_ws)

    """
    MAIN UPDATING
    """
    states = ["susceptible", "exposed", "infectious_asymptomatic", "infectious_presymptomatic", "infectious_symptomatic", "recovered", "dead"]
    print("t, #" + ", #".join(states))
    results = np.empty([len(T_ARRAY), len(states)])

    for n, _t in enumerate(T_ARRAY):
        # update one time-step
        update(_t)
        # Store Results
        results[n,:] = observe(agents)
        print(_t, results[n,:])

    '''
    Plotting
    '''
    title = r"$N_{Agents}$" + r"={:d}, $N_e(t=0)$={:d}, Net: Watts-Strogatz ({:d}, {:.2f}) seed={:d}".format(
        N_AGENTS, N_INFECTED_INIT, k_ws, p_ws, _seed)
    filename = "AggregateResults_N-{:d}-{:d}_WS({:d},{:.0e})_seed{:d}".format(
        N_AGENTS, N_INFECTED_INIT, k_ws, p_ws, _seed)
    plot_statistics(T_ARRAY, results, states, title=title, filename=FOLDER + filename)

    return results

# %% [markdown]
# # MAIN: do simulations

# %%
'''
RUN
'''
seed = 2     # Set random seed, so that running the model again with the same seed results in the same output.
agents = []     # Will be loaded with a list of agents
network = {}        # Will be loaded with the network, i.e. with Nodes and Links.
k_ws = 6
p_ws = 0.1
results = run(seed, k_ws = k_ws, p_ws = p_ws)




# %%
"""
ADVANCED:
Plot R0
"""
"""
 plot_r_values(agents, fname=FOLDER+"R0values_N-{:d}-{:d}_WS({:d},{:.0e})_seed{:d}".format(N_AGENTS, N_INFECTED_INIT, k_ws, p_ws, seed))
"""

# %% [markdown]
# # Extra Task: 
# How many contacts do agents need to have on average to trigger outbreaks?

# %%

""" 
ADVANCED:     
#   - CHECK DIFFERENT k OF WATTS_STROGATZ MODEL 
#   - PERFORM STATISTICAL RUNS / ENSMEBLE RUNS
#   - AND PLOT 
#       (1) HOW OFTEN OUTBREAKS OCCUR, 
#       (2) THE FRACTION OF DEATHS/INFECTED IN SUCH OUTBREAKS
#       OVER THE USED k VALUES
# 
"""
"""
final_fractions_networks = []
k_ws_list = [2, 4, 6, 8, 10]
p_ws = 0.1
for k_ws in k_ws_list:
    ensemble_results = []
    seeds = np.arange(1, 11)
    for seed in seeds:
        results = run(seed, k_ws=k_ws, p_ws=p_ws)
        #  append the final state of the system (to our ensemble results) and discard the rest of the dynamcis
        ensemble_results.append(results[-1, :])   

    ensemble_results = np.array(ensemble_results)
    #print(ensemble_results)
    outbreaks = np.where(ensemble_results[:, 0] < 0.5 * N_AGENTS)[0]   # outbreak? If more than 50% infected.
    fraction_outbreaks = len(outbreaks) / len(seeds)
    if len(outbreaks) > 0:
        fraction_infected_in_outbreak = 1 - np.mean([ensemble_results[outbreak, 0] / N_AGENTS for outbreak in outbreaks])
        fraction_dead_in_outbreak = np.mean([ensemble_results[outbreak, -1] / N_AGENTS for outbreak in outbreaks])
    else:
        fraction_infected_in_outbreak = 0
        fraction_dead_in_outbreak = 0

    final_fractions_networks.append([fraction_outbreaks, fraction_infected_in_outbreak, fraction_dead_in_outbreak])

final_fractions_networks = np.array(final_fractions_networks)
fractions_outbreaks = final_fractions_networks[:, 0]
fractions_infected_in_outbreak = final_fractions_networks[:, 1]
fractions_dead_in_outbreak = final_fractions_networks[:, 2]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.plot(k_ws_list, fractions_outbreaks, lw=3, label="Fraction Outbreaks", color="b")
ax.legend(loc="upper left")
ax2 = ax  # .twinx()
ax2.plot(k_ws_list, fractions_infected_in_outbreak, lw=3, label=r"Fraction Infected in Outbreak $P(e|outbreak)$",
            color="g")
ax2.legend(loc="lower left")
ax3 = ax.twinx()
ax3.plot(k_ws_list, fractions_dead_in_outbreak, lw=3, label=r"Fraction Dead in Outbreak $P(d|outbreak)$", color="k")
ax3.legend(loc="center right")
fig.tight_layout()
plt.savefig(
    FOLDER + "Fractions_WS(p={:.0e})_varied-k_N-{:d}-{:d}.pdf".format(p_ws, N_AGENTS, N_INFECTED_INIT),
    bbox_inches="tight")
plt.show()
"""
"""
"""

# %%



