# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
from create_networks import *
plt.rcParams.update({"font.size": 20})
from const import *

agents = np.nan
N_subgroups = [np.nan for _ in range(3)]


def plot_r_values(n_convolve=20, title=None):
    """ Plot all r values over time, via convolution """
    global agents
    all_rs = np.array([ag.r for ag in agents])
    all_tes = np.array([ag.t_e for ag in agents])
    al_rs_without_nan = all_rs[np.where(not np.isnan(all_rs))]
    al_tes_without_nan = all_tes[np.where(not np.isnan(all_tes))]
    inds = np.argsort(al_tes_without_nan)
    rs_sort = al_rs_without_nan[inds]
    rs_conv = np.convolve(rs_sort, np.ones(N_convolve, ) / n_convolve, mode='valid')
    times_conv = np.convolve(al_tes_without_nan[inds], np.ones(n_convolve, ) / n_convolve, mode="valid")
    plt.title("R- Values")
    plt.plot(times_conv, rs_conv)
    fig.tight_layout()
    if title is not None:
        plt.savefig(title+".pdf", bbox_inches=tight)
    plt.show()
    return


def plot_network(g, title=None):
    """ Plot a network and its adjacency Matrix"""
    global agents, N_subgroups
    indices_groups = [
        np.arange(N_subgroups[0]),  # children
        np.sum(N_subgroups[:1]) + np.arange(N_subgroups[1]),  # adults
        np.sum(N_subgroups[:2]) + np.arange(N_subgroups[2])  # risk
    ]

    colors_edges = [
        (255/255, 255/255, 255/255),
        (127/255, 201/255, 127/255),
        (190/255, 174/255, 212/255),
        (253/255, 192/255, 134/255),
        (0, 0, 0),

    ]
    colors_edges_dict = dict(zip(np.arange(5), colors_edges))
    cmap_edges = mpl.colors.ListedColormap(colors_edges[:])
    # cmap_edges_reduced = mpl.colors.ListedColormap(colors_edges[1:])
    colors_nodes = [
        (228/255, 26/255, 28/255),
        (55/255, 126/255, 184/255),
        (77/255, 175/255, 74/255)
    ]
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_edges = mpl.colors.BoundaryNorm(bounds, cmap_edges.N)
    # cNorm = mpl.colors.Normalize(vmin=0, vmax=3)
    # cmap_edges = mpl.cm.

    cmap_nodes = mpl.colors.ListedColormap(colors_nodes)

    # groups = dict(zip([ag.id for ag in agents], [ag.group for ag in agents]))
    for ag, n in zip(agents, g.nodes):
        g.nodes[n]["group"] = ag.group
        g.nodes[n]["id"] = ag.group
        g.nodes[n]["state"] = 0

    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0, 0])

    edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())

    pos = nx.spring_layout(g, k=0.2)

    edge_color = [colors_edges_dict[weight] for weight in list(weights)]
    nx.draw(g, pos, ax=ax, node_color=[g.nodes.data("group")[n] for n in g.nodes], cmap=cmap_nodes,
            edgelist=edges, edge_color=edge_color)  # , norm_edges=norm_edges)
    # plt.colorbar()
    ax.set_title("Network: "+title)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, len(agents))
    ax.set_xlim(0, len(agents))
    # ax.spy(nx.adjacency_matrix(g), ms=)
    ax.matshow(nx.adjacency_matrix(g).todense(), cmap=cmap_edges, norm=norm_edges)
    ax.set_title("Corresponding Adjacency Matrix")
    N_subgroups = [len(group) for group in indices_groups]
    ax.vlines(np.cumsum(N_subgroups), 0, len(agents), linestyles="--", lw=1, alpha=0.4, colors="gray")
    ax.hlines(np.cumsum(N_subgroups), 0, len(agents), linestyles="--", lw=1, alpha=0.4, colors="gray")
    fig.tight_layout()
    if title is not None:
        plt.savefig(title+".pdf", bbox_inches=tight)
    plt.show()
    return


def plot_dists(title=None):
    fig = plt.figure(figsize=(16, 9))
    gs = mpl.gridspec.GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(AGENT_INFECTIOUSNESS.rvs(1000), density=True)
    ax1.set_title("Agent infectiousness")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(INCUBATION_TIME_DIST.rvs(1000), density=True)
    ax1.set_title("Incubation Time")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(TIME_ONSET_TO_DEATH.rvs(1000), density=True)
    ax1.set_title("Time from onset to death")

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.hist(RECOVERY_TIME_SYMPTOMATIC.rvs(1000), density=True)
    ax1.set_title("Recovery Time")

    ax2 = fig.add_subplot(gs[1, 2])
    # symptomatic
    ax2.set_xlim(0, 20)
    # ax2.set_ylim(0,1)
    for t in range(10):
        ag = Agent()
        inc = INCUBATION_TIME_DIST.rvs()
        # ag.infectious_period = [inc - START_INFECTION_BEFORE_SYMPTOM,
        #                   inc + START_INFECTION_BEFORE_SYMPTOM + RECOVERY_TIME_SYMPTOMATIC.rvs()]
        ag.symptomatic = True
        ag.t_e = 0
        ag.t_max_infectiousness = inc
        ag.infectiousness = AGENT_INFECTIOUSNESS.rvs()
        ts = np.arange(ag.t_e, ag.t_max_infectiousness + 9)
        i_s = []
        for t_ in ts:
            i_s.append(ag.get_infectiousness(t_))
        ax2.plot(ts, i_s, alpha=1, color="k")
        ax2.set_title("Infectiousness Symptomatic")

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.set_xlim(0, 20)
    # ax1.set_ylim(0,1)
    # symptomatic
    for t in range(10):
        ag = Agent()
        inc = INCUBATION_TIME_DIST.rvs()
        # ag.infectious_period = [inc - START_INFECTION_BEFORE_SYMPTOM,
        #                   inc + START_INFECTION_BEFORE_SYMPTOM + RECOVERY_TIME_SYMPTOMATIC.rvs()]
        ag.symptomatic = False
        ag.t_e = 0
        ag.t_max_infectiousness = inc
        ag.infectiousness = AGENT_INFECTIOUSNESS.rvs()
        ts = np.arange(ag.t_e, ag.t_max_infectiousness + 9, step=0.5)
        i_s = []
        for t_ in ts:
            i_s.append(ag.get_infectiousness(t_))
        ax1.plot(ts, i_s, alpha=1, color="k")
    ax1.set_title("Infectiousness A-Symptomatic")
    fig.tight_layout()
    if title is not None:
        plt.savefig(title+".pdf", bbox_inches=tight)
    plt.show()
    return


def plot_statistics(results, policy_dates, title=None):
    t_array = results[:, 0]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    state_labels = ["s", "e", "i_a", "i_ps", "i_s", "d", "r_a", "r_s"]
    ax.plot(t_array, results[:, 0 + 1], label=state_labels[0])
    ax.set_ylim(ymax=np.sum(results[0, 1:]))

    plt.legend()
    ax2 = ax.twinx()
    for n, state in enumerate(state_labels[1:]):
        ax2.plot(t_array, results[:, n+1+1], label=state)

    ymin, ymax = ax.get_ylim()
    for pol, pol_date in policy_dates:
        ax.vlines(pol_date, ymin, ymax, linestyles="--", color="gray")
    ax2.set_ylim(0, )
    plt.legend()
    fig.tight_layout()
    if title is not None:
        plt.savefig(title + ".pdf", bbox_inches=tight)
    plt.show()
    return
