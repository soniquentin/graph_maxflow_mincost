#author: Quentin LAO
#date: 2023/10/22

import pandas as pd
import matplotlib.pyplot as plt
import heapq
import numpy as np

def parse_graph(edge_graph = 'test_graph_edges.csv', node_graph = 'test_graph_nodes.csv'):

    """
    EDGES
    """

    #import the csv file
    df_edges = pd.read_csv(edge_graph)
    #CSV of form (node1, node2, cost, capacity)

    edges = {}

    #iterate through the rows of the csv
    for index, row in df_edges.iterrows():
        #create a tuple of the two nodes
        edge = (row['node1'], row['node2'])
        #add the edge to the dictionary of edges
        edges[edge] = {'flow' : 0, 'capacity': int(row['capacity']), 'cost': int(row['cost'])}


    """
    NODES
    """

    #import the csv file
    df_nodes = pd.read_csv(node_graph)
    nodes_from_csv = set(df_nodes['node'].tolist())

    #get the list of nodes from edges
    node_list = set(df_edges['node1'].tolist() + df_edges['node2'].tolist())

    #check that the nodes in the node graph are a subset of the nodes in the edge graph
    assert node_list.issubset(nodes_from_csv)

    nodes = {}

    #iterate through the rows of the csv
    for index, row in df_nodes.iterrows():
        #add the node to the dictionary of nodes
        nodes[row['node']] = {'x' : float(row['x']), 'y': float(row['y']), 'path_from_s': [0, row['node']]}

    return edges, nodes


def display_graph(G:tuple, step = 0):
    edges, nodes = G

    #create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #plot the nodes
    for node in nodes:
        ax.scatter(nodes[node]['x'], nodes[node]['y'], color = 'black')
        #Add the node number
        ax.text(nodes[node]['x'], nodes[node]['y'] - 0.1, f"{node}", fontsize = 8, color = 'grey')
        #Add the shortest from s
        if nodes[node]['path_from_s'] != None:
            ax.text(nodes[node]['x'], nodes[node]['y'] - 0.2, f"{nodes[node]['path_from_s'][0]}", fontsize = 8, color = 'green')

    #plot the edges
    treated_edges = []
    for edge in edges:
        node1, node2 = edge
        treated_edges.append((node1, node2))
        x1, y1 = nodes[node1]['x'], nodes[node1]['y']
        x2, y2 = nodes[node2]['x'], nodes[node2]['y']

        #plot the edge with arrow
        lamb = 0.90
        if (node2, node1) in treated_edges:
            mu = 0.2
            norm_vect = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            theta = np.arctan((y2 - y1)/(x2 - x1))

            inflection_x = x2 + (norm_vect/2)*np.cos(theta) + mu*np.sin(theta)
            inflection_y = y2 + (norm_vect/2)*np.sin(theta) - mu*np.cos(theta)

            #Plot horizontal line
            ax.plot([x1, inflection_x], [y1, inflection_y], color = 'black')
            ax.arrow(inflection_x, inflection_y, lamb*(x2 - inflection_x), lamb*(y2 - inflection_y), head_width = 0.05, color = 'black')
            ax.text(inflection_x, inflection_y, f"{edges[edge]['flow']}/{edges[edge]['capacity']}", fontsize = 8, color = 'red')
            ax.text(inflection_x, inflection_y - 0.1, f"{edges[edge]['cost']}", fontsize = 8, color = 'blue')
        else:
            ax.arrow(x1, y1, lamb*(x2 - x1), lamb*(y2 - y1), head_width = 0.05, color = 'black')
            ax.text((x1 + x2)/2, (y1 + y2)/2, f"{edges[edge]['flow']}/{edges[edge]['capacity']}", fontsize = 8, color = 'red')
            ax.text((x1 + x2)/2, (y1 + y2)/2 - 0.1, f"{edges[edge]['cost']}", fontsize = 8, color = 'blue')
        

    #Add the step number as a title
    ax.set_title(f"Step {step}")

    plt.show()



def get_R_star(G:tuple) :
    edges, nodes = G

    edges_R = {}

    for edge in edges:
        node1, node2 = edge
        if edges[edge]['flow'] < edges[edge]['capacity']:
            edges_R[(node1,node2)] = {'flow' : 0, 'capacity' : edges[edge]['capacity'] - edges[edge]['flow'] , 'cost' : edges[edge]['cost']}
        if edges[edge]['flow'] > 0:
            edges_R[(node2,node1)] = {'flow' : 0, 'capacity' : edges[edge]['flow'] , 'cost' : -edges[edge]['cost']}

    return edges_R, nodes



def add_flow(G:tuple, path_flow:list, flow:int):
    edges, nodes = G

    #iterate through the path
    for i in range(len(path_flow) - 1):
        edge = (path_flow[i], path_flow[i + 1])
        #Check capacity
        assert edges[edge]['flow'] + flow <= edges[edge]['capacity'], f"Flow exceeds capacity on edge {edge}"

        edges[edge]['flow'] += flow
    
    return edges, nodes



def shortest_path_from_s(G:tuple):
    """
    Update the path_from_s in nodes and return edges, nodes with nodes updated. path_from_s is the length of the shortest path from s to the node

    Bellman-ford
    """
    edges, nodes = G


    # Initialize distances to all nodes as infinity except for the source node.
    distance = {node: float('inf') for node in nodes}
    distance["s"] = 0

    for _ in range(len(nodes) - 1):
        for edge in edges:
            node1, node2 = edge
            cost_edge = edges[edge]['cost']
            if distance[node1] != float('inf') and distance[node1] + cost_edge < distance[node2]:
                distance[node2] = distance[node1] + cost_edge
                nodes[node2]['path_from_s'] = [distance[node2], node1, edges[edge]['capacity']]

    #Check for negative cycles
    for edge in edges:
        node1, node2 = edge
        cost_edge = edges[edge]['cost']
        if distance[node1] != float('inf') and distance[node1] + cost_edge < distance[node2]:
            raise Exception("Negative cycle detected")

    return edges, nodes



def get_path_from_s(G:tuple, node:str):
    edges, nodes = G
    possible_add_flow = float('inf')

    path = []
    path.append(node)

    while node != 's':
        possible_add_flow = min(possible_add_flow, nodes[node]['path_from_s'][2])
        node = nodes[node]['path_from_s'][1]
        path.append(node)

    path.reverse()
    return path, possible_add_flow




if __name__ == '__main__':
    step =  0

    #Initialize the graph
    G = parse_graph(edge_graph = 'exo_edges.csv', node_graph = 'exo_nodes.csv')
    R_star = shortest_path_from_s(get_R_star(G))
    path_flow, possible_add_flow = get_path_from_s(R_star, 't')
    print(f"[STEP {step}] shortest path in R*(f) : {' --> '.join(path_flow)}\n")
    display_graph(R_star, step)

    while True :
        step += 1
        G = add_flow(G, path_flow, possible_add_flow)
        R_star = shortest_path_from_s(get_R_star(G))
        previous_flow = [i for i in path_flow] #copy the list
        path_flow, possible_add_flow = get_path_from_s(R_star, 't')
        if previous_flow == path_flow:
            break
        print(f"[STEP {step}] shortest path in R*(f) : {' --> '.join(path_flow)}\n")
        display_graph(R_star, step)


    display_graph(G, "G FINAL")





