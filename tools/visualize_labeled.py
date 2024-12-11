import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from math import isnan

def network_dict(genome, state, nodes, conns):
    network = genome.network_dict(state, nodes, conns)
    network["nodes_data"] = nodes
    return network  

def visualize_labeled(
        genome,
        network,
        activation_functions=None,
        rotate=0,
        reverse_node_order=False,
        size=(1000, 1000, 1000),
        color=("yellow", "white", "blue"),
        with_labels=False,
        edgecolors="k",
        arrowstyle="->",
        arrowsize=8,
        edge_color=(0.3, 0.3, 0.3),
        save_path="network.svg",
        save_dpi=800,
        **kwargs,
    ):

        conns_list = list(network["conns"])
        input_idx = genome.get_input_idx()
        output_idx = genome.get_output_idx()

        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }
        if reverse_node_order:
            topo_order = topo_order[::-1]

        G = nx.DiGraph()

        if not isinstance(size, tuple):
            size = (size, size, size)
        if not isinstance(color, tuple):
            color = (color, color, color)

        activation_functions_dict = {i: act for i, act in enumerate(activation_functions)}
        activation_functions_dict[-1] = "ID"

        custom_labels = {}
        for node in topo_order:
            if node in input_idx:
                G.add_node(node, subset=node2layer[node], size=size[0], color=color[0])
            elif node in output_idx:
                G.add_node(node, subset=node2layer[node], size=size[2], color=color[2])
            else:
                G.add_node(node, subset=node2layer[node], size=size[1], color=color[1])
            #print(network["nodes_data"][node])
            if activation_functions is not None and not isnan(network["nodes_data"][node][4]) \
                and node not in input_idx and node not in output_idx:
                act_name = activation_functions_dict[int(network["nodes_data"][node][4])]
                custom_labels[node] = f"{node}\n{act_name}"
            else:
                custom_labels[node] =  f"{node}"

        edge_widths = []
        edge_colors = []
        for conn in network["conns"].values():
            G.add_edge(conn['in'], conn['out'], weight=conn['weight'].item())
            edge_widths.append(max(abs(conn['weight'].item()), 0.2))
            edge_colors.append("red" if conn['weight'].item() < 0 else "blue")
        pos = nx.multipartite_layout(G)

        def rotate_layout(pos, angle):
            angle_rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
            rotated_pos = {}
            for node, (x, y) in pos.items():
                rotated_pos[node] = (
                    cos_angle * x - sin_angle * y,
                    sin_angle * x + cos_angle * y,
                )
            return rotated_pos

        rotated_pos = rotate_layout(pos, rotate)

        node_sizes = [n["size"] for n in G.nodes.values()]
        node_colors = [n["color"] for n in G.nodes.values()]

        plt.clf()
        nx.draw(
            G,
            pos=rotated_pos,
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=with_labels,
            edgecolors=edgecolors,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            edge_color=edge_colors,
            width=edge_widths,
            labels=custom_labels,
            font_size=6,
            **kwargs,
        )
        plt.savefig(save_path, dpi=save_dpi)