# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import networkx as nx
import seaborn

plt.rcParams.update({"font.size": 20})


def plot_net(g, agents, filename=None):
    """
    Plot the network of g.
    Each risk group has a certain color
    :param g: a network graph consisting of indices of all the agents (stored in the list agents), links between them.
    :param agents: list of all agents
    :return:
    """
    # From https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3
    # An excellent source if you want to define well-fitting, visible and commonly used colors/-schemes.
    # I choose
    #   Number=3,
    #   qualitative nature (maximum difference between the colors)
    colors_nodes = [
        (228 / 255, 26 / 255, 28 / 255),  # Red = RiskGroup 0
        (55 / 255, 126 / 255, 184 / 255),  # Blue = RiskGroup 1
        (77 / 255, 175 / 255, 74 / 255)  # Green = RiskGroup 2
    ]
    # From the colors, define colormap:
    cmap_nodes = mpl.colors.ListedColormap(colors_nodes)

    # For each node in the graph or agent in the list, add the group in the node_color array
    # Alternative:
    #   You can also plot the different infection states in a colormap of 9 colors.
    #   In this case you would simply add the state instead of the group as an attribute to the node
    #   and then plot these attributes
    node_colors = []
    for ag, n in zip(agents, g.nodes):
        node_colors.append(ag.group)
        # g.nodes[n]["group"] = ag.group
    # node_colors = [g.nodes.data("group")[n] for n in g.nodes]

    # Determine a good layout for the nodes (clustering close links)
    pos = nx.spring_layout(g, k=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw(g, pos, ax=ax, node_color=node_colors, cmap=cmap_nodes, node_size=100)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    # Get the node-degrees for each node (how many links does each node have)
    total_node_degrees = nx.adjacency_matrix(g).sum(axis=1)
    print("An agent has on average : {:.2f}+-{:.2f} links in the net".format(total_node_degrees.mean(),
                                                                             total_node_degrees.std()))
    return


def plot_statistics(results, states, title="", filename=None):
    """
    Plots the aggregate results over time
    :param results: Array with dim (len(t_array), len(states)+1):
            Column 0: time
            Column 1-9: aggregate numbers in this state at the corresponding time
    :param states: ["s", "e", "i_a", "i_ps", "i_s", "r_a", "r_s", "d"]
    :param title: (string) The title added to the axis of the figure.
    :param filename: (string) Name of saved pdf figure in the folder specified by FOLDER
    :return:
    """
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(results[:, 0][0], results[:, 0][-1])
    colors_lines = np.array([
        (27, 158, 119),
        (217, 95, 2),
        (117, 112, 179),
        (231, 41, 138),
        (102, 166, 30),
        (230, 171, 2),
        (166, 118, 29),
        (102, 102, 102),
    ]) / 255
    ax.plot(results[:, 0], results[:, 1], "--", lw=3, label="s", color=colors_lines[0])
    ax.set_ylim(0, )
    plt.legend(loc="center left")
    ax2 = ax.twinx()
    ax2.set_ylim(0, np.max(results[:, 2:6]) * 1.1)
    for n, state in enumerate(states[1:]):
        ax2.plot(results[:, 0], results[:, n + 2], lw=3, label=state, color=colors_lines[n + 1, :])
    plt.legend(loc="center right")
    ax.set_title(title)
    if filename is not None:
        fig.tight_layout()
        plt.savefig(filename + ".pdf", bbox_inches='tight')
    plt.show()
    return
