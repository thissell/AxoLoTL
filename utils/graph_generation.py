import numpy as np
import torch
import dgl

def gen_graph(num_nodes, top_k=5):
    fully_connected = torch.ones([num_nodes, num_nodes], dtype=torch.float).nonzero(as_tuple=True)
    g = dgl.graph(fully_connected, num_nodes = num_nodes)

    nf = torch.tensor(np.random.randint(16, size=num_nodes))

    inc_m = g.inc('both').to_dense().numpy()
    L = inc_m @ inc_m.T

    e, u = np.linalg.eigh(L)
    e, u = torch.from_numpy(e), torch.from_numpy(u)
    _, topk = torch.topk(e.real, k=top_k)

    topk_indices = topk.unsqueeze(0).expand(u.size(0), -1)
    out = torch.gather(u.real, 1, topk_indices)

    return g, e, u, nf, out, num_nodes

def create_nonsense_batch(num_nodes, num_graphs, device, top_k=5):
    e_list = []
    u_list = []
    g_list = []
    nf_list = []
    pe_list = []
    l_list = []

    for _ in range(num_graphs):
        g, e, u, nf, pe, l = gen_graph(num_nodes, top_k=top_k)
        e_list.append(e)
        u_list.append(u)
        g_list.append(g)
        nf_list.append(nf)
        pe_list.append(pe)
        l_list.append(l)

    e_batch = torch.stack(e_list).to(device)
    u_batch = torch.stack(u_list).to(device)
    g_batch = dgl.batch(g_list).to(device)
    nf_batch = torch.stack(nf_list).to(device)
    pe_batch = torch.stack(pe_list).to(device)
    l_batch = torch.tensor(l_list).to(device)

    return g_batch, e_batch, u_batch, nf_batch, pe_batch, l_batch