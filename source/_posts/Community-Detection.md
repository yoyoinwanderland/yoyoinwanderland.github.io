---
title: Community Detection in Python
date: 2017-08-08 19:48:15
tags: 
- Python Packages
- Network Analytics
- Machine Learning
category: 
- 时习之
- Machine Learning
description: A study note for performing community detection in Python using networkX and iGraph
---
## NetworkX vs. IGraph

### Implementation

* NetworkX: implemented in Python
* IGraph: implemented in C

### Performance

IGraph wins.

|                             | IGraph                                   | NetworkX                                 |
| --------------------------- | ---------------------------------------- | ---------------------------------------- |
| Single-source shortest path | 0.012 s                                  | 0.152 s                                  |
| PageRank                    | 0.093 s                                  | 3.949 s                                  |
| K-core                      | 0.022 s                                  | 0.714 s                                  |
| Minimum spanning tree       | 0.044 s                                  | 2.045 s                                  |
| Betweenness                 | 946.8 s (edge) + 353.9 s (vertex) (~ 21.6 mins) | 32676.4 s (edge)  22650.4 s (vertex) (~15.4 hours) |

  See [ref](https://graph-tool.skewed.de/performance)

###  No. of Community Detection Algorithms

IGraph wins.

* NetworkX: only optimal modularity.
* IGraph: nine algorithms including optimal modularity; edge betweenness etc.

### Ease of Programming

NetworkX wins.

#### Load a graph

* NetworkX can simply load a graph from a list of edge tuples.

```
edges = [(1,2),(2,3),(3,4),(4,5)]
G = nx.Graph()

# add edges from txt 
G.add_edges_from(edges)
```

* IGraph needs to first load vertices and then a list of edge tuples

```
# create graph
g = Graph()

# in order to add edges, we have to add all the vertices first

#iterate through edges and put all the vertices in a list
vertex = []
for edge in edges:
    vertex.extend(edge)

g.add_vertices( list( set(vertex))) # add a list of unique vertices to the graph
g.add_edges(edges) # add the edges to the graph. 
"""
Note: add_edges is much quicker than add_edge. for every time add_edge is performed, the entire data structure in C has to be renewed with a new series ordering.
"""

print g
```

####  Get information from existing graph

* NetworkX

```
## get no. of neighbors of a node by node name
node_index = G.vs.index(target_node) 
neighbors = G.adjacency_list()[node_index]
```

* IGraph

```
## get no. of neighbors of a node by node name
node_index = g.vs.find(target_node).index  #find index by nodename
neighbors = []

# g.incident(node_index) gives a series of edges that passes the target node index
# iterate through the edges, and append the other node into neighbors list

for edge in g.es[g.incident(node_index)]: 
	if edge.target == node_index:
		neighbors.append(g.vs[edge.source]["name"])
    else:
    	neighbors.append(g.vs[edge.target]["name"])
```



## Community Detection Algorithms

A list of algorithms available in IGraph include:

* [Optimal Modularity](http://tuvalu.santafe.edu/~aaronc/modularity/)
* [Edge Betweenness (2001)](https://arxiv.org/abs/cond-mat/0112110)
* [Fast Greedy (2004)](https://arxiv.org/abs/cond-mat/0408187)
* [Walktrap (2005)](https://arxiv.org/abs/physics/0512106)
* [Eigenvectors (2006)](https://arxiv.org/abs/physics/0605087)
* [Spinglass (2006)](https://arxiv.org/abs/cond-mat/0603718)
* [Label Propagation (2007)](https://arxiv.org/abs/0709.2938)
* [Multi-level (2008)](https://arxiv.org/abs/0803.0476)
* [Info Map (2008)](https://arxiv.org/abs/0906.1405)


### Summary

* For directed graph: go with **Info Map**. Else, pls continue to read.

* If compuational resources is not a big problem, and the graph is < 700 vertices & 3500 edges, go with **Edge Betweenness**; it yields the best result.

* If cares about modularity, any of the remaining algorithms will apply;

  * If the graph is particularly small: < 100 vertices, then go with **optimal modularity**;
  * If you want a first try-on algorithm, go with **fast greedy** or **walktrap**
  * If the graph is bigger than 100 vertices and not a de-generated graph, and you want something more accurate than fast greedy or walktrap, go with **leading eigenvectors**
  * If you are looking for a solution that is similar to K-means clustering, then go for **Spinglass**

  ​


### Optimal Modularity

* **Definition of modularity**: 

   > Modularity compares the number of edges inside a cluster with the expected number of edges that one would find in the cluster if the network were a random network with the same number of nodes and where each node keeps its degree, but edges are otherwise randomly attached.

   Modularity is a measure of the segmentation of a network into partitions. The higher the modularity, the denser in-group connections are and the sparser the inter-group connections are. 

* **Methodology**: 

  GNU linear programming kit.

* **Evaluation**: 
  * Better for smaller communities with less than 100 vertices for the reasons of implementation choice.
  * Resolution limit: when the network is large enough, small communities tend to be combined even if they are well-shaped.



### Fast Greedy

* **Methodology**:

  Bottom up hierarchical decomposition process. It will merge two current communities iteratively, with the goal to achieve the maximum modularity gain at local optimal.

* **Evaluation**: 

  * Pretty fast, can merge a sparse graph at linear time.
  * Resolution limit: when the network is large enough, small communities tend to be combined even if they are well-shaped.

### Walktrap

* **Methodology**:

  Similar to fast greedy. It is believed that when we walk some random steps, it is large likely that we are still in the same community as where we were before. This method firstly performs a random walk 3-4-5, and merge using modularity  with methods similar to fast greedy.

* **Evaluation**:

  * A bit slower than fast greedy;
  * A bit more accurate than fast greedy.

### Multi-level

* **Methodology**:

  Similar to fast greedy, just that nodes are not combined, they move around communities to make dicision if they will contribute to the modularity score if they stay.

### Eigenvectors

- **Methodology**:

  A top down approach that seeks to maximize modularity. It concerns decomposing a modularity matrix. 

- **Evaluation**:

  - More accurate than fast greedy
  - Slower than fast greedy
  - Limitation: not stable on degenerated graphs (might not work!)

### Label Propogation

* **Methodology**:

  A bit like k-clustering, with initialization k different points. It uses an iterative method (again just like k-means): the target label will be assigned with the most "vote" of the lables from its neighbors; until the current label is the most frequent label.


* **Evaluation**:
  * Very fast
  * Like K-Means, random initialization yields different results. Therefore have to run multiple times (suggested 1000+) to achieve a consensus clustering.

### Edge Betweenness

- **Definition of edge betweenness**:

  > Number of shortest path that passes the edge.

  It's not difficult to imagin that, if there is an edge that connects two different groups, then that edge will has to be passed through multiple times when we count the shortest path. Therefore, by removing the edge that contains with the highest number of shortest path, we are disconnecting two groups.

- **Methodology**:

  Top down hierarchical decomposition process.

- **Evalution**:

  - Generally this approach gives the most satisfying results from my experience.
  - Pretty slow method. The computation for edge betweenness is pretty complex, and it will have to be computed again after removing each edge. Suitable for graph with less than 700 vertices and 3500 edges.
  - It produces a dendrogram with no reminder to choose the appropriate number of communities. (But for IGraph it does a function that output the optimal count for a dendrogram).

### Spinglass

* **Methodology**:

  Complicated enough for me to ignore..

* **Evaluation**:

  * Not fast
  * has a bunch of hyperparameters to tune from

### Infomap

* **Methodology**:

  > It is based on information theoretic principles; it tries to build a grouping which provides the shortest description length for a random walk on the graph, where the description length is measured by the expected number of bits per vertex required to encode the path of a random walk.

* **Evaluation**:

  * Used for directed graph analytics





## Reference

* [Graph-tool performance comparison](https://graph-tool.skewed.de/performance)
* [Python IGraph Manual](http://igraph.org/python/doc/igraph.Graph-class.html)
* [Modularity (Networks)](https://en.wikipedia.org/wiki/Modularity_(networks))
* [Tamas' answer to "What are the differences between community detection algorithms?"](https://stackoverflow.com/questions/9471906/what-are-the-differences-between-community-detection-algorithms-in-igraph/)