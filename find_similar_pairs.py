
import os
import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from itertools import combinations

class Graph():
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self,u,v):
        self.graph[u].append(v)

    def isCyclicUtil(self, v, visited, recStack):

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node,visited,recStack) == True:
                    return True
        return False

def topologicalSortUtil(v):
    global Stack, visited, adj
    visited[v] = True

    # Recur for all the vertices adjacent to this vertex
    # list<AdjListNode>::iterator i
    for i in adj[v]:
        if (not visited[i[0]]):
            topologicalSortUtil(i[0])

    # Push current vertex to stack which stores topological
    # sort
    Stack.append(v)

# The function to find longest distances from a given vertex.
# It uses recursive topologicalSortUtil() to get topological
# sorting.
def longestPath(s):
    global Stack, visited, adj, V
    dist = [-10**9 for i in range(V)]

    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for i in range(V):
        if (visited[i] == False):
            topologicalSortUtil(i)
    # print(Stack)

    # Initialize distances to all vertices as infinite and
    # distance to source as 0
    dist[s] = 0
    # Stack.append(1)

    # Process vertices in topological order
    while (len(Stack) > 0):

        # Get the next vertex from topological order
        u = Stack[-1]
        del Stack[-1]
        #print(u)

        # Update distances of all adjacent vertices
        # list<AdjListNode>::iterator i
        if (dist[u] != 10**9):
            for i in adj[u]:
                # print(u, i)
                if (dist[i[0]] < dist[u] + i[1]):
                    dist[i[0]] = dist[u] + i[1]

    # Prthe calculated longest distances
    # print(dist)
    rets = []
    for i in range(V):
        if dist[i] == -10**9:
            ret = "INF"
        else:
            ret = dist[i]
        rets.append(ret)
        # print("INF ", end="") if (dist[i] == -10**9) else print(dist[i], end=" ")
    return rets

sources_by_sequence = {
    "Chopstick_Using": 15,
    "Hand_Drawing": 4,
    "Sonic_Drawing": 15,
    "Dough_Rolling": 26,
    "Suturing": 14,
    "Needle_Passing": 16,
    "Knot_Tying":15
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', '--sequence', required=True,
        choices=["Chopstick_Using", "Hand_Drawing", "Sonic_Drawing",
                 "Dough_Rolling", "Suturing", "Needle_Passing", "Knot_Tying"])
    args = parser.parse_args()
    source = sources_by_sequence[args.sequence]

    directory = os.path.join("EPIC-Skills2018/annotations", args.sequence)
    frames_by_dir = {"Chopstick_Using": "ChopstickUsing", "Dough_Rolling": "DoughRolling",
                     "Hand_Drawing": "HandDrawing"}

    print("[INFO] Directory: {}".format(directory))
    consistent_df = pd.DataFrame()
    files = glob(os.path.join(directory, "splits", "*.csv"))
    for file in files:
        df = pd.read_csv(file).iloc[:, :2]
        consistent_df = pd.concat([consistent_df, df])

    consistent_df.drop_duplicates(inplace=True)

    all_videos = os.listdir(
        os.path.join("frames",
                     frames_by_dir.get(args.sequence, args.sequence)))
    count = len(all_videos)
    all_pairs = list(combinations(all_videos, 2))
    print("[INFO] #Videos: {}".format(count))
    print("[INFO] #Max Pairs: {}".format(len(all_pairs)))
    inconsistent_pairs = []
    consistent_pairs = []

    g = Graph(count)
    for idx, row in consistent_df.iterrows():
        g.addEdge(row['Better'], row['Worse'])
        consistent_pairs.append((row['Better'], row['Worse']))

    print("[INFO] #Cons. Pairs: {}".format(len(consistent_pairs)))
    print("[INFO] %Cons. Pairs: {:.2f}".format(len(consistent_pairs)/len(all_pairs)*100))

    if g.isCyclic() == 1:
        print("Graph has a cycle")
        exit()
    else:
        print("Graph has no cycle")

    for pair in all_pairs:
        if (pair not in consistent_pairs) and (pair[::-1] not in consistent_pairs):
            inconsistent_pairs.append(pair)

    V, Stack, visited = count, [], [False for i in range(count+1)]
    adj = [[] for i in range(count+1)]

    mapp = dict(enumerate(all_videos))
    mapp_inv = {j:i for i,j in mapp.items()}

    for idx, row in consistent_df.iterrows():
        adj[mapp_inv[row['Better']]].append([mapp_inv[row['Worse']], 1])

    # for row in all_pairs:
    #     adj[mapp_inv[row[0]]].append([mapp_inv[row[1]], 1])
    #     adj[mapp_inv[row[1]]].append([mapp_inv[row[0]], 1])

    dists = longestPath(source)

    similar_pairs = []
    for pair in inconsistent_pairs:
        if dists[mapp_inv[pair[0]]] == "INF" or dists[mapp_inv[pair[1]]] == "INF":
            continue
        if abs(dists[mapp_inv[pair[0]]] - dists[mapp_inv[pair[1]]]) <= 1:
            similar_pairs.append(pair)

    print("[INFO] #Sim. Pairs: {} with source: {}".format(len(similar_pairs), source))
    print("[INFO] #Sim. Pairs: {:.2f}".format(len(similar_pairs)/len(all_pairs)*100))
    similar_df = pd.DataFrame(similar_pairs, columns=['Better', 'Worse'])
    similar_df.to_csv(os.path.join(directory, "similar_pairs.csv"), index=False)
