# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
from create_networks import *
plt.rcParams.update({"font.size":20})
from const import *

def plot_r_values(agents, N_convolve=20):
    all_rs = np.array([ag.r for ag in agents])
    all_tes = np.array([ag.t_e for ag in agents])
    al_rs_without_nan = all_rs[np.where(np.isnan(all_rs) == False)]
    al_tes_without_nan = all_tes[np.where(np.isnan(all_tes) == False)]
    inds = np.argsort(al_tes_without_nan)
    rs_sort = al_rs_without_nan[inds]
    rs_conv = np.convolve(rs_sort, np.ones(N_convolve, ) / N_convolve, mode='valid')
    times_conv = np.convolve(al_tes_without_nan[inds], np.ones(N_convolve, ) / N_convolve, mode="valid")
    plt.title("R- Values")
    plt.plot(times_conv, rs_conv)
    plt.show()
    return


def plot_network(g, agents, N_subgroups, title):
    """ Plot a network and its adjacency Matrix"""

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
        (0,0,0),

    ]
    colors_edges_dict = dict(zip(np.arange(5), colors_edges))
    cmap_edges = mpl.colors.ListedColormap(colors_edges[:])
    #cmap_edges_reduced = mpl.colors.ListedColormap(colors_edges[1:])
    colors_nodes = [
        (228/255, 26/255, 28/255),
        (55/255, 126/255, 184/255),
        (77/255, 175/255, 74/255)
    ]
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_edges = mpl.colors.BoundaryNorm(bounds, cmap_edges.N)
    #cNorm = mpl.colors.Normalize(vmin=0, vmax=3)
    #cmap_edges = mpl.cm.

    cmap_nodes = mpl.colors.ListedColormap(colors_nodes)

    #groups = dict(zip([ag.id for ag in agents], [ag.group for ag in agents]))
    for ag, n in zip(agents, g.nodes):
        g.nodes[n]["group"] = ag.group
        g.nodes[n]["id"] = ag.group
        g.nodes[n]["state"] = 0

    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=(2,1))
    ax = fig.add_subplot(gs[0, 0])

    edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())

    pos = nx.spring_layout(g, k=0.2)

    edge_color = [colors_edges_dict[weight] for weight in list(weights)]
    nx.draw(g, pos, ax=ax, node_color=[g.nodes.data("group")[n] for n in g.nodes], cmap=cmap_nodes,
            edgelist=edges, edge_color=edge_color)#, norm_edges=norm_edges)
    #plt.colorbar()
    ax.set_title("Network: "+title)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, len(agents))
    ax.set_xlim(0, len(agents))
    #ax.spy(nx.adjacency_matrix(g), ms=)
    ax.matshow(nx.adjacency_matrix(g).todense(), cmap=cmap_edges, norm = norm_edges)
    ax.set_title("Corresponding Adjacency Matrix")
    N_subgroups = [len(group) for group in indices_groups]
    ax.vlines(np.cumsum(N_subgroups), 0, len(agents), linestyles="--", lw=1, alpha=0.4, colors="gray")
    ax.hlines(np.cumsum(N_subgroups), 0, len(agents), linestyles="--", lw=1, alpha=0.4, colors="gray")
    fig.tight_layout()
    plt.show()
    return


