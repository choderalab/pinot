from pinot.app.experiment import Train, Test
import torch
import copy


class MultitaskTrain(Train):
    """ Training experiment when heads need to be masked for each task.
    Attributes
    ----------
    data : generator
        training data, containing a tuple of mask, graphs, y-labels
    net : `pinot.Net` object
        model with parameters to be trained
    record_interval: int, default=1
        interval at which `states` are being recorded
    optimizer : `torch.optim.Optimizer` object, default=`torch.optim.Adam(1e-5)`
        optimizer used for training
    n_epochs : int, default=100
        number of epochs
    """

    def __init__(self, net, data, optimizer, n_epochs=100, record_interval=1):
        super(MultitaskTrain, self).__init__(
            net=net,
            data=data,
            optimizer=optimizer,
            n_epochs=n_epochs,
            record_interval=record_interval,
        )


    def train(self):
        """ Train the model for multiple steps and
        record the weights once every
        `record_interval`.

        """
        from tqdm import tqdm
        for epoch_idx in tqdm(range(int(self.n_epochs))):
            self.train_once()

            # if a new regressor was added, add optimizer param group
            for _, output_regressor in self.net.output_regressors.items():
                try:
                    self.update_optimizer(output_regressor)
                except ValueError:
                    pass

            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        self.states["final"] = copy.deepcopy(self.net.state_dict())

        if hasattr(self.optimizer, "expectation_params"):
            self.optimizer.expectation_params()

        return self.net


    def update_optimizer(self, output_regressor):
        """
        Adds parameters from a specific regressor to the optimizer.
        """
        lr = self.optimizer.param_groups[0]['lr']
        if hasattr(output_regressor, 'likelihood'):
            params = [{'params': output_regressor.hyperparameters(), 'lr': lr * 1e-3},
                      {'params': output_regressor.variational_parameters()},
                      {'params': output_regressor.likelihood.parameters()}]
        else:
            params = [{'params': output_regressor.parameters()}]
        
        for p in params:
            self.optimizer.add_param_group(p)

# class MultitaskTest(Test)