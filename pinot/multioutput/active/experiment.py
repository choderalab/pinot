from pinot.active.experiment import BayesOptExperiment
from torch.utils.data import WeightedRandomSampler


class MultitaskBayesOptExperiment(BayesOptExperiment):
    """ Implements active learning experiment with multiple task target.
    """

    def __init__(self, *args, **kwargs):

        super(MultitaskBayesOptExperiment, self).__init__(*args, **kwargs)

    def acquire(self):
        """ Acquire new training data.
        """
        # set net to eval
        self.net.eval()

        # split input target
        gs, ys = self.new_data

        scores = []
        # for each task
        for task in range(ys.size(1)):

            # if we trained for this task
            if str(task) in self.net.output_regressors:

                # get the predictive distribution
                distribution = self.net.condition(gs, task)

                # workup
                distribution = self.workup(distribution)

                # get score
                score = self.acquisition(
                    distribution, y_best=self.y_best[task]
                )
                epsilon = 1e-4 * torch.rand(score.shape)

                if torch.cuda.is_available:
                    epsilon = epsilon.cuda()

                scores.append(score + epsilon)

        # harmonize the scores
        # TODO: AVERAGE SCORES
        # first, scale the scores
        scaled_scores = [(s - s.mean()) / s.std() for s in scores]
        # next: stack the scores
        scaled_scores = torch.stack(scaled_scores)
        # next: average the scores
        score = scaled_scores.mean(axis=0)

        if not self.weighted_acquire:
            # argmax
            best = torch.topk(score, self.q).indices
        else:
            # generate probability distribution
            weights = torch.exp(-score)
            weights = weights / weights.sum()
            best = WeightedRandomSampler(
                weights=weights, num_samples=self.q, replacement=False
            )
            best = torch.IntTensor(list(best))

        # pop from the back so you don't disrupt the order
        best = best.sort(descending=True).values
        self.old.extend([self.new.pop(b) for b in best])

    def update_data(self):
        """ Update the internal data using old and new.
        """
        # grab new data
        self.new_data = self.slice_fn(self.data, self.new)

        # grab old data
        self.old_data = self.slice_fn(self.data, self.old)

        # set y_max
        gs, ys = self.old_data

        # get the max value observed for each task
        # TODO: Do this intelligently.
        y_best = -10 * torch.ones(ys.size(1))

        for idx, col in enumerate(ys.T):
            col_labeled = col[~torch.isnan(col)]
            if col_labeled.size(0):
                y_best[idx] = torch.max(col_labeled)

        self.y_best = y_best
