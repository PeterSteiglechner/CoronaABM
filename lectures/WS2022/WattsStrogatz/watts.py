import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
G = nx.watts_strogatz_graph(10, 4, 0)
pos = nx.spring_layout(G)
nx.draw(G, pos = pos)
plt.show()


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
G = nx.Graph()
for _ in range(10):
    G.add_node(_)
# pos = nx.spring_layout(G)
nx.draw(G, pos = pos)
plt.show()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
G = nx.watts_strogatz_graph(10, 4, 0.1)
#pos = nx.spring_layout(G)
nx.draw(G, pos = pos)
plt.show()


