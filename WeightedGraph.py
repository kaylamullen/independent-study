from PriorityQueue import PriorityQueue
import numpy as np

class WeightedGraph:
    def __init__(self):
        self.graph = {}
        self.all_edges = {}
        self.cars_on_edge = {}

    def addNode(self, node):
        if node not in self.graph:
            self.graph[node] = {}

    def addEdge(self, node1, node2, weight):
        if node1 not in self.graph or node2 not in self.graph:
            self.addNode(node1)
            self.addNode(node2)
        self.graph[node1][node2] = weight
        self.cars_on_edge[(node1, node2)] = 0
        self.all_edges[(node1, node2)] = weight

    def modifyWeight(self, node1, node2, weight):
        weight = int(weight)
        if node1 not in self.graph or node2 not in self.graph:
            raise ValueError("Both nodes must exist")
        if node2 not in self.graph[node1]:
            raise ValueError("Edge does not exist")
        self.graph[node1][node2] = weight
        self.all_edges[(node1, node2)] = weight

    def getMatrix(self):
        m = np.zeros((109, 109))
        # print(m.shape)
        for node1, edges in self.graph.items():
            for node2, weight in edges.items():
                m[node1][node2] = weight
        return m
        

    def getNeighbors(self, node):
        if node not in self.graph:
            raise ValueError("Node does not exist")
        return self.graph[node]

    def get_edge(self, node1, node2):
        return self.graph[node1][node2]

    def getNodes(self):
        return list(self.graph.keys())
    
    def path_to_node(self, end_node, prev_node, start_node, path):
        if end_node == start_node:
            path.append(start_node)
            return path
        else:
            path.append(end_node)
            return self.path_to_node(prev_node[end_node], prev_node, start_node, path)

    def dijkstra_shortest_path(self, start_node, end_node):
        nodes = self.getNodes()
        # lists the node that found the curr node as the val and curr node as key
        prev_node = {}
        d = {}
        pq = PriorityQueue()
        for node in nodes:
            d[node] = -1
        s = start_node
        d[s] = 0
        while s != end_node:
            neighbors = self.getNeighbors(s)
            for v in neighbors.keys():
                if d[v] == -1:
                    pq.insert(v, neighbors[v])
                    d[v] = neighbors[v] + d[s]
                    prev_node[v] = s
                if neighbors[v]+d[s] < d[v]:
                    pq.insert(v, neighbors[v]+d[s])
                    d[v] = neighbors[v] + d[s]
                    prev_node[v] = s
            try:
                new, val = pq.extractMin()
                # prev_node[new] = s
                s = new
            except IndexError:
                print(f"No path between {start_node} and {end_node}")
                return [], -1

        path = self.path_to_node(end_node, prev_node, start_node, [])
        path = path[::-1]
        return path, d[end_node]
    
def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()

    g = WeightedGraph()

    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")

    # Use the edge information to populate your graph object
        g.addNode(int(edge[0]))
        g.addNode(int(edge[1]))
        g.addEdge(int(edge[0]), int(edge[1]), int(edge[2]))



    # Close the file safely after done reading
    file.close()
    return g
    
if __name__ == "__main__":
    graph = read_graph('project_files/grid100.txt')
    print(graph.dijkstra_shortest_path(0, 3))