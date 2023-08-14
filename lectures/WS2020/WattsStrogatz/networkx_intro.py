import networkx as nx
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(2,2))
G = nx.Graph()
G.add_node("Ahmad")
G.add_node("Beate")
G.add_node("Can")
G.add_node("Deborah")
G.add_edges_from([("Ahmad", "Beate"), ("Ahmad", "Can"), ("Ahmad", "Deborah"), ("Can", "Deborah")])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="gold")
#fig.tight_layout()
x_values, y_values = zip(*pos.values())
x_max = max(x_values)
x_min = min(x_values)
x_margin = (x_max - x_min) * 0.5
plt.xlim(x_min - x_margin, x_max + x_margin)

plt.savefig("network.pdf")#, bbox_inches="tight")
plt.show()
print(nx.adjacency_matrix(G).todense())
print(G.adj)
