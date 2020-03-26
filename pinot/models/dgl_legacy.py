""" Legacy models from DGL.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import math
import dgl
from dgl.nn import pytorch as dgl_pytorch

# =============================================================================
# MODULE CLASS
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
            self,
            in_feat,
            out_feat,
            model_name='SAGEConv',
            kwargs={'aggregator_type': 'mean'}):
        super(GN, self).__init__()
        self.gn = getattr(
            dgl_pytorch.conv,
            model_name)(in_feat, out_feat, **kwargs)

    def forward(self, g):
        x = g.ndata['h']
        x = self.gn(g, x)
        g.ndata['h'] = x
        return g
