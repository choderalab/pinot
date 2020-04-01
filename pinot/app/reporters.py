# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import dgl
import torch
from abc import ABC
import numpy as np
import time

# =============================================================================
# MODULE CLASSES
# =============================================================================
class ReporterBase(ABC):

    def before(self, *args, **kwargs):
        pass

    def during(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

class MarkdownReporter(ReporterBase):
    def __init__(self, path, ds_tr, ds_te, args, net):
        self.path = path
        self.ds_tr = ds_tr
        self.ds_te = ds_te
        self.args = args
        self.net = net 

    def before(self):
        import time
        from datetime import datetime
        import os
        
        losses = np.array([0.])
        self.time0 = time.time()

        f_handle = open(self.path + '/report.md', 'w')
        f_handle.write(self.path)
        f_handle.write('\n')
        f_handle.write('===========================')
        f_handle.write('\n')
        f_handle.write('# Model Summary\n')
        for arg in vars(self.args):
            f_handle.write(arg+ '=' + str(getattr(self.args, arg)))
            f_handle.write('\n')

        f_handle.write(str(self.net))
        f_handle.write('\n')

        self.f_handle = f_handle

    def after(self, net):
        loss_tr_this_epoch = [torch.mean(net.loss(g, y)) for g, y in self.ds_tr]
        loss_te_this_epoch = [torch.mean(net.loss(g, y)) for g, y in self.ds_te]
        
        f_handle = self.f_handle
        time1 = time.time()
        f_handle.write('# Time used\n')
        f_handle.write(str(time1 - self.time0) + ' s\n')

        f_handle.write('# Performance \n')
        f_handle.write('{:<15}'.format('|'))
        f_handle.write('{:<15}'.format('|NLL')+ '|' + '\n')

        f_handle.write('{:<15}'.format('|' + '-' * 13))
        f_handle.write('{:<15}'.format('|' + '-' * 13))
        f_handle.write('|' + '\n')

        f_handle.write('{:<15}'.format('|TRAIN'))
        f_handle.write('{:<15}'.format('|%.2f' % loss_tr_this_epoch[-1]) + '|' + '\n')

        f_handle.write('{:<15}'.format('|TEST'))
        f_handle.write('{:<15}'.format('|%.2f' % loss_te_this_epoch[-1]) + '|' + '\n')

        f_handle.close()


class VisualReporter(ReporterBase):
    def __init__(self, path, ds_tr, ds_te):
        self.path = path
        self.ds_tr = ds_tr
        self.ds_te = ds_te
        self.losses_tr = []
        self.losses_te = []

    def during(self, net):
        from matplotlib import pyplot as plt
        plt.style.use('fivethirtyeight')

        net.eval()
        loss_tr_this_epoch = [torch.mean(net.loss(g, y)) for g, y in self.ds_tr]
        loss_te_this_epoch = [torch.mean(net.loss(g, y)) for g, y in self.ds_te]
 
        self.losses_tr.append(torch.mean(torch.stack(loss_tr_this_epoch)).detach().numpy())
        self.losses_te.append(torch.mean(torch.stack(loss_te_this_epoch)).detach().numpy())

        plt.figure()
        plt.plot(self.losses_tr, label='training')
        plt.plot(self.losses_te, label='test')
        plt.legend()
        plt.savefig(self.path + '/loss.png')

    def after(self, net):
        np.save(self.path + '/losses_tr.npy', np.array(self.losses_tr))
        np.save(self.path + '/losses_te.npy', np.array(self.losses_te))

class WeightReporter(ReporterBase):
    def __init__(self, path):
        self.path = path
        self.idx = 0
        
    def during(self, net):
        torch.save(net.state_dict(), self.path + '/w' + str(self.idx) + '.ds')
        self.idx += 1
