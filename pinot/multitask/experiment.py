from pinot.app.experiment import Train
import torch


class MultiTaskTrain(Train):
    """Training experiment when heads need to be masked for each task."""

    def __init__(self, net, data, optimizer, n_epochs=100, record_interval=1):
        super(MultiTaskTrain, self).__init__(
            net=net,
            data=data,
            optimizer=optimizer,
            n_epochs=n_epochs,
            record_interval=record_interval,
        )

    def train_once(self):
        """Train the model for one batch."""
        for l, g, y in self.data:

            def l():
                """ """
                loss = self.net.loss(g, y)
                self.optimizer.zero_grad()
                loss.backward()
                return loss

            self.optimizer.step(l)
