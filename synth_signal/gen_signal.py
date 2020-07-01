import numpy as np
import networkx as nx
from synth_signal.road_network import RoadNetwork
import random


def sample_itinerary_periodic(num_nodes, num_sampled_trajectories):
    itineraries = []
    for t, num_traj in enumerate(num_sampled_trajectories):
        itin_t = []
        for i in range(int(num_traj)):
            source, target = np.random.choice(a=num_nodes, size=2, replace=False)
            itin_t.append((source, target))
        itineraries.append(itin_t)

    return itineraries


def sample_trajectories_run(t, road_net, periodic_itineraries,
                            num_sampled_trajectories, regulator=1):
    t = t % len(num_sampled_trajectories)
    num_trajectories = int(np.floor(num_sampled_trajectories[t] * regulator))
    allowed_nodes = [node for node, i in enumerate(road_net.congested_nodes) if i == 0]
    allowed_periodic_itineraries = [itin for itin in periodic_itineraries[t] if
                                    ((itin[0] in allowed_nodes) and (itin[1] in allowed_nodes))]

    if len(allowed_periodic_itineraries) > num_trajectories:
        itineraries_idx = np.random.choice(a=len(allowed_periodic_itineraries),
                                           size=num_trajectories, replace=False)
        itineraries = [allowed_periodic_itineraries[i] for i in itineraries_idx]
    else:
        itineraries = allowed_periodic_itineraries

    trajectories = [road_net.shortest_paths[(itin[0], itin[1])] for itin in itineraries]

    for i in range(len(allowed_periodic_itineraries) // 5):
        source, target = np.random.choice(a=allowed_nodes, size=2, replace=False)
        trajectories.append(road_net.shortest_paths[(source, target)])

    return trajectories


def generate_signal(G, period=200, signal_length=2000):

    assert list(G.nodes()) == list(range(len(G))), f"\nGraph nodes should be named from 0 to len(G)-1\n" \
                                                   f"Run this code to rename the graph nodes:\n\n" \
                                                   f"node_list = list(G.nodes()) \n" \
                                                   f"for i, old_id in enumerate(node_list): \n" \
                                                   f"  map_labels = {{}} \n" \
                                                   f"  map_labels[old_id] = i \n" \
                                                   f"  nx.relabel_nodes(G, map_labels, copy=False)"

    current_trajectories = []
    road_net = RoadNetwork(G)
    avg_path_length = nx.average_shortest_path_length(G)
    num_nodes = road_net.num_nodes
    avg_capacity = np.mean([item[1] for item in road_net.capacities.items()])
    speeds = []
    times = []
    traffics = []
    time_list = np.linspace(0, 2 * np.pi, period)
    num_sampled_trajectories = np.floor(
        0.15 * num_nodes * avg_capacity / avg_path_length + 0.6 * num_nodes * avg_capacity / avg_path_length * np.sin(
            time_list) ** 4)
    periodic_itineraries = sample_itinerary_periodic(num_nodes, num_sampled_trajectories)

    for t in range(signal_length):

        if t % period == 0:
            regulator = random.uniform(0.6, 1.0)

        new_trajectories = sample_trajectories_run(t, road_net, periodic_itineraries,
                                                   num_sampled_trajectories, regulator)

        current_trajectories = current_trajectories + new_trajectories

        speed, traffic, current_trajectories = road_net.update_routine(current_trajectories)

        speeds.append(speed)
        traffics.append(traffic)
        times.append((t % period) / period)

    return speeds, traffics, times
