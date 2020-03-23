""" Chain mutiple layers of GN together.
"""

class Sequential(torch.nn.Module):
    def __init__(self, model, config, feature_units=117, input_units=128):
        super(Sequential, self).__init__()

        dim = input_units
        self.exes = []

        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units),
            torch.nn.Tanh())

        def apply_atom_in_graph(fn):
            def _fn(g):
                g.apply_nodes(
                    lambda node: {'h': fn(node.data['h'])}, ntype='atom')
                return g
            return _fn

        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except:
                pass

            if type(exe) == int:
                setattr(
                    self,
                    'd' + str(idx),
                    model(dim, exe))

                dim = exe
                self.exes.append('d' + str(idx))

            elif type(exe) == str:
                activation = getattr(torch.nn.functional, exe)

                setattr(
                    self,
                    'a' + str(idx),
                    apply_atom_in_graph(activation))

                self.exes.append('a' + str(idx))

            elif type(exe) == float:
                dropout = torch.nn.Dropout(exe)
                setattr(
                    self,
                    'o' + str(idx),
                    apply_atom_in_graph(dropout))

                self.exes.append('o' + str(idx))

            self.readout = ParamReadout(
                readout_units=readout_units,
                in_dim=dim)

    def forward(self, g, return_graph=False):

        g.apply_nodes(
            lambda nodes: {'h': self.f_in(nodes.data['h0'])},
            ntype='atom')

        for exe in self.exes:
            g = getattr(self, exe)(g)

        g = self.readout(g)

        if return_graph == True:
            return g

        g = hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(
            g)

        g = hgfp.mm.energy_in_heterograph.u(g)

        u = torch.sum(
                torch.cat(
                [
                    g.nodes['mol'].data['u' + term][:, None] for term in [
                        'bond', 'angle', 'torsion', 'one_four', 'nonbonded', '0'
                ]],
                dim=1),
            dim=1)


        return u
