import random
import networkx as nx
import numpy as np


class RoadNetwork(object):

    def __init__(self, G, weight=None, traffic=None, speed=None,
                 capacities=None, cap_limits=(20, 41),
                 congested_nodes=None, congestion_threshold=5,
                 accident_nodes=None, accident_length=10, accident_recurr=300):

        if nx.is_directed(G):
            self.G = G.to_undirected()
        else:
            self.G = G
        # self.G = self.G.to_directed()

        self.num_nodes = len(self.G)

        if weight is None:
            self.edge_weights = {}
            for edge in self.G.edges():
                self.edge_weights[edge] = 1.0
        else:
            self.edge_weights = weight

        nx.set_edge_attributes(self.G, self.edge_weights, 'edge_weights')
        self.shortest_paths = {}
        for i in self.G.nodes():
            for j in self.G.nodes():
                if i != j:
                    self.shortest_paths[(i, j)] = nx.shortest_path(self.G, i, j,
                                                                   weight='edge_weights')

        if capacities is None:
            cap_low, cap_up = cap_limits
            self.capacities = {}
            values = np.random.randint(cap_low, cap_up, self.num_nodes)
            for i, cap in zip(G.nodes(), values):
                self.capacities[i] = cap

        if traffic is None:
            traffic = np.zeros(self.num_nodes)
        self.traffic = traffic

        if speed is None:
            speed = np.ones(self.num_nodes)
        self.speed = speed

        if congested_nodes is None:
            congested_nodes = np.zeros(self.num_nodes)
        self.congested_nodes = congested_nodes
        self.congestion_threshold = congestion_threshold

        if accident_nodes is None:
            accident_nodes = np.zeros(self.num_nodes)
        self.accident_nodes = accident_nodes
        self.accident_length = accident_length
        self.accident_times = self.accident_length * np.ones(self.num_nodes)
        self.accident_recurr = accident_recurr

    def update_speed(self):
        curr_speed = []
        for node, traf in zip(self.G.nodes(), self.traffic):
            cap = self.capacities[node]
            if traf < cap:
                curr_speed.append(1)
            else:
                curr_speed.append(max(1 / 3 * (4 - traf / cap), 0.05))

        for node, acc in enumerate(self.accident_nodes):
            if acc == 1:
                curr_speed[node] = 0.00001

        self.speed = np.array(curr_speed)

    def gen_accident(self):
        num = random.uniform(0, 1)
        if num < (1 / self.accident_recurr):
            non_accident_nodes = [node for node, acc in enumerate(self.accident_nodes) if acc == 0]
            new_accident_node = random.choice(non_accident_nodes)
            self.accident_nodes[new_accident_node] = 1

    def update_accident(self):
        curr_accident = [node for node, acc in enumerate(self.accident_nodes) if acc == 1]
        for node in curr_accident:
            if self.accident_times[node] == 0:
                self.accident_nodes[node] = 0
                self.accident_times[node] = self.accident_length

        for node, acc in enumerate(self.accident_nodes):
            if acc == 1:
                self.accident_times[node] -= 1

    def update_shortest_paths(self):
        node_weight = [1 / sp for sp in self.speed]
        for edge in self.G.edges():
            self.edge_weights[edge] = node_weight[edge[0]] + node_weight[edge[1]]
        nx.set_edge_attributes(self.G, self.edge_weights, 'edge_weights')

        for i in self.G.nodes():
            for j in self.G.nodes():
                if i != j:
                    self.shortest_paths[(i, j)] = nx.shortest_path(self.G, i, j,
                                                                   weight='edge_weights')

    def update_congested_nodes(self):
        for node in self.G.nodes():
            if self.accident_nodes[node] ==1 or self.traffic[node] >= self.capacities[node] * self.congestion_threshold:
                self.congested_nodes[node] = 1
            else:
                self.congested_nodes[node] = 0

    def update_traffic(self, current_trajectories):
        traffic = np.zeros(self.num_nodes)
        num_traj = len(current_trajectories)
        adv_prob = np.random.random(num_traj)

        for idx, (traj, prob) in enumerate(zip(current_trajectories, adv_prob)):
            traffic[traj[0]] += 1
            if self.speed[traj[0]] > prob:
                if (len(traj)) == 1:
                    current_trajectories[idx] = []
                elif (self.congested_nodes[traj[1]] == 0) or (self.congested_nodes[traj[0]] == 1):
                    current_trajectories[idx] = traj[1:]

        while [] in current_trajectories:
            current_trajectories.remove([])

        self.traffic = traffic
        return current_trajectories

    def update_routine(self, current_trajectories):
        current_trajectories = self.update_traffic(current_trajectories)
        self.gen_accident()
        self.update_accident()
        self.update_speed()
        self.update_congested_nodes()
        self.update_shortest_paths()

        return self.speed, self.traffic, current_trajectories

    def get_speed(self):
        return self.speed

    def get_traffic(self):
        return self.traffic

    def get_shortest_paths(self):
        return self.shortest_paths

    def get_congested_nodes(self):
        return self.congested_nodes
