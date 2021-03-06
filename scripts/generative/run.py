import pinot
import dgl
import torch
import numpy as np

def run():
    # construct the vgae object
    vgae = pinot.generative.VGAE(units=32)

    # grab some data
    ds_tr = pinot.data.esol()[:10]

    # discard the measurement
    gs, _ = zip(*ds_tr)

    # batch the graphs into one large one
    g = dgl.batch(gs)

    # get the adjacency matrix for the giant graph
    a = torch.tensor(g.adjacency_matrix()).to_dense()

    # initialize a representation net
    layer = pinot.representation.dgl_legacy.gn(
            model_name='GraphConv',
            kwargs={})

    net = pinot.representation.Sequential(
        layer,
        config=[32, 'tanh', 32, 'tanh'])


    opt = torch.optim.Adam(
            list(net.parameters())\
          + list(vgae.parameters()),

          1e-2)

    # train
    for _  in range(3000):
        opt.zero_grad()
        x = net(g, return_graph=True).ndata['h']
        loss = vgae.loss(x, a, sample_shape=[64]).sum()
        print(loss)
        loss.backward()
        opt.step()

    a = vgae.generate(x)

    g = pinot.generative.utils.graph_from_adjacency_matrix(a)

    print(g)


if __name__ == '__main__':
    run()
