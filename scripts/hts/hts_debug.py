import pinot

def squared_error(net, distribution, y, *args, n_samples=16, batch_size=32, **kwargs):
    """ Squared error. """
    y_hat = distribution.sample().detach().cpu().reshape(-1, 1)
    return (y - y_hat)**2

def test(
    net,
    data,
    states,
    sampler=None,
    metrics=[pinot.rmse, pinot.r2, pinot.avg_nll, squared_error],
    record_interval=50
    ):
    """ Test experiment. Metrics are applied to the saved states of the
    model to characterize its performance.


    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        Metrics used to characterize the performance.

    Methods
    -------
    test : Run the test experiment.

    """
    import torch
    
    def compute_conditional(net, data, batch_size):
        # compute conditional distribution in batched fashion
        # and remove off-diagonals of covariance (ensure independence)
        locs, scales = [], []
        for idx, d in enumerate(data.batch(batch_size, partial_batch=True)):

            g_batch, _ = d
            distribution_batch = net.condition(g_batch)
            loc_batch = distribution_batch.mean.flatten().cpu()
            scale_batch = distribution_batch.variance.pow(0.5).flatten().cpu()
            locs.append(loc_batch)
            scales.append(scale_batch)

        distribution = torch.distributions.normal.Normal(
            loc=torch.cat(locs),
            scale=torch.cat(scales)
        )
        return distribution

    # switch to test
    net.eval()

    # initialize an empty dict for each metrics
    results = {}

    for metric in metrics:
        results[metric.__name__] = {}

    # make g, y into single batches
    g, y = data.batch(len(data))[0]
    for state_name, state in states.items():  # loop through states
        
        if state_name % record_interval == 0:

            net.load_state_dict(state)

            if net.has_exact_gp:
                batch_size = len(data)
            else:
                batch_size = 32

            # compute conditional distribution in batched fashion
            distribution = compute_conditional(net, data, batch_size)
            y = y.detach().cpu().reshape(-1, 1)
            for metric in metrics:  # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric(
                        net,
                        distribution,
                        y,
                        sampler=sampler,
                        batch_size=batch_size
                    ).detach().cpu().numpy()
                )

    return results




def run(args):
    import pinot
    import torch
    import os
    import logging
    import time
    import pickle

    # Specify accelerator (if any)
    device = torch.device("cuda:0" if args.cuda else "cpu:0")

    # If output folder doesn't exist, create a new one
    try:
        os.mkdir('bin')
        os.mkdir(args.output)
    except:
        pass

    logging.basicConfig(
        filename=os.path.join(args.output, args.log),
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )
    logging.debug(args)

    seed = args.seed
    savefile_no_seed = args.states.split('dict_state')[1].replace('seed=None.p', '')
    savefile = (f'{savefile_no_seed}_seed={seed}')

    print(savefile)
    logging.debug("savefile = {}".format(savefile))

    #############################################################################
    #
    #                   LOAD DATA
    #
    #############################################################################

    start = time.time()

    path = f'./bin/mpro_hts_{args.sample_frac[0]}_seed={seed}.bin'
    
    if os.path.isfile(path):
        # see if we've already serialized it
        data = pinot.data.datasets.Dataset()
        data = data.load(path)
    else:
        # otherwise, load from scratch
        data = pinot.data.mpro_hts(sample_frac=args.sample_frac[0], seed=seed)
        data.save(path)

    # move to cuda
    data = data.to(device)
    
    # Set batch size and log
    batch_size = args.batch_size
    end = time.time()
    logging.debug("Finished loading all data after {} seconds".format(end-start))


    #############################################################################
    #
    #                   TRAIN A SUPERVISED MODEL FIRST
    #
    #############################################################################

    def parse_states(states_file):
        """ Parse the net architecture from the states file.

        """
        # parse filename
        _, meta = states_file.split('dict_state_')
        (reg, n_layers, n_units, unit_type, activation,
         n_epochs, batch_size, _, split, frac,
         anneal_factor, n_inducing, normalize, index, _) = [
            m.split('=')[-1] 
            for m in meta.split('_')
        ]

        # clean up string into correct data structure
        integerize = lambda x: int(x.replace('x', ''))
        n_layers = integerize(n_layers)
        n_units = integerize(n_units)
        n_inducing = integerize(n_inducing)

        architecture = [unit_type, n_units, 'activation', activation]*n_layers

        return architecture, n_inducing


    def get_net_and_optimizer(architecture, n_inducing_points):
        """

        """
        representation = pinot.representation.sequential.SequentialMix(
            architecture,
        )

        output_regressor = pinot.regressors.VariationalGaussianProcessRegressor

        # First train a fully supervised Net to use as Baseline
        net = pinot.Net(
            representation=representation,
            output_regressor_class=output_regressor,
            n_inducing_points=n_inducing_points
        )
        net.to(device)
        return net

    # get states
    states = pickle.load(open(f'./{args.output}/{args.states}', 'rb'))

    # make appropriately sized network
    architecture, n_inducing_points = parse_states(args.states)
    supNet = get_net_and_optimizer(architecture, n_inducing_points)

    start = time.time()
    
    # get results
    debug_results = test(
        supNet,
        data,
        states,
        metrics=[pinot.rmse, pinot.r2, pinot.avg_nll, squared_error]
    )

    end = time.time()
    
    # logging and write to file
    logging.debug("Finished testing net after {} seconds and save state dict".format(end-start))
    pickle.dump(debug_results, open(f'./{args.output}/debug_{savefile}.p', 'wb'))


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser("HTS supervised learning")

    parser.add_argument(
        '--states',
        type=str,
        default="None",
        help="Name of file with network states"
    )

    parser.add_argument(
        '--cuda',
        action="store_true",
        default=False,
        help="Using GPU"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="out",
        help="Name of folder to store results"
    )

    parser.add_argument(
        '--log',
        type=str,
        default="logs",
        help="Log file"
    )

    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        '--sample_frac',
        nargs="+",
        type=float,
        default=0.005, # 0.1
        help="Proportion of dataset to use"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Setting the seed for random sampling"
    )

    args = parser.parse_args()

    run(args)