from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch

from lib.context_free_grammar import UnaryNode, BinaryNode, Node, id_to_str_dict

def create_graph(g: dgl.DGLGraph, current_node: Node, parent_node_id: int = -1, global_node_id: int = -1):
    g.add_nodes(1, {'feat': torch.tensor([current_node.token.embed], dtype=torch.int)})
    
    my_node_id = global_node_id + 1

    node_list = [[my_node_id, current_node]]

    new_node_id = my_node_id

    if parent_node_id != -1:
        g.add_edges(parent_node_id, my_node_id)

    if isinstance(current_node, UnaryNode):
        new_node_id, more_nodes = create_graph(g, current_node.child, parent_node_id=my_node_id, global_node_id=new_node_id)
        node_list += more_nodes
    elif isinstance(current_node, BinaryNode):
        new_node_id, more_nodes1 = create_graph(g, current_node.left, parent_node_id=my_node_id, global_node_id=new_node_id)
        node_list += more_nodes1

        new_node_id, more_nodes2 = create_graph(g, current_node.right, parent_node_id=my_node_id, global_node_id=new_node_id)
        node_list += more_nodes2
    
    return new_node_id, node_list


def to_graph(node: Node):
    g = dgl.graph(([], []))
    _, node_list = create_graph(g, node)
    return g, node_list


def draw_tree(g: dgl.graph):
    G = dgl.to_networkx(g, ['feat'])
    plt.figure(figsize=[7,7])

    pos = graphviz_layout(G, prog="dot")
    labels = nx.get_node_attributes(G,'feat')
    lb_str = {i: id_to_str_dict[labels[i].item()] for i in labels}
    nx.draw(G, pos, with_labels=True, labels=lb_str, node_color="white")

    plt.show()