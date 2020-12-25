def squared_error(net, distribution, y, *args, n_samples=16, batch_size=32, **kwargs):
    """ Squared error. """
    y_hat = distribution.sample().detach().cpu().reshape(-1, 1)
    return (y - y_hat)**2

def test(
    net,
    data,
    states,
    sampler=None,
    metrics=[pinot.rmse, pinot.r2, squared_error]
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
    from tqdm import tqdm
    
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
    for state_name, state in tqdm(states.items()):  # loop through states
        
        if state_name % 100 == 0:

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

    layer_type = args.architecture[0]
    n_layers = len(args.architecture) // 4
    n_units = args.architecture[1]
    activation = args.architecture[3]

    seed = args.seed
    savefile = (f'reg={args.regressor_type}_a={n_layers}x_{n_units}x'
                f'_{layer_type}_{activation}_n={args.n_epochs}_b={args.batch_size}'
                f'_wd={args.weight_decay}_lsp={args.label_split[0]}_frac={args.sample_frac}'
                f'_anneal={args.annealing}_induce={args.n_inducing_points}_normalize={args.normalize}'
                f'_{args.index}_seed={seed}')

    print(savefile)
    logging.debug("savefile = {}".format(savefile))

    #############################################################################
    #
    #                   LOAD DATA
    #
    #############################################################################

    start = time.time()

    path = f'./bin/mpro_hts_{args.sample_frac[0]}.bin'
    
    if os.path.isfile(path):
        # see if we've already serialized it
        data = pinot.data.datasets.Dataset()
        data = data.load(path)
    else:
        # otherwise, load from scratch
        data = getattr(pinot.data, args.data)(sample_frac=args.sample_frac[0])
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
        _, meta = t.split('dict_state_')
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
        n_inducing_points = integerize(n_inducing_points)

        architecture = [unit_type, n_units, 'activation', activation]*n_layers

        return architecture, n_inducing_points


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
    states = pickle.load(open(args.states, 'rb'))

    # make appropriately sized network
    architecture, n_inducing_points = parse_states(args.states)
    supNet = get_net_and_optimizer(architecture, n_inducing_points)

    start = time.time()
    
    # mini-batch if we're using variational GP
    data = data.batch(batch_size)

    # get results
    debug_results = test(
        supNet,
        data,
        states,
        metrics=[pinot.rmse, pinot.r2, squared_error]
    )

    end = time.time()
    
    # logging and write to file
    logging.debug("Finished testing net after {} seconds and save state dict".format(end-start))
    pickle.dump(debug_results, open(f'./{args.output}/debug_{savefile}_seed={seed}.p', 'wb'))


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
        '--weight_decay',
        default=0.01,
        type=float,
        help="Weight decay for optimizer",
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
        '--label_split',
        nargs="+",
        type=list,
        default=[4, 1],
        help="Training-testing split for labeled data"
    )
    parser.add_argument(
        '--index',
        type=int,
        default=1,
        help="Arbitrary index to append to logs"
    )
    parser.add_argument(
        '--annealing',
        type=float,
        default=1.0,
        help="Scaling factor on the KL term in the variational inference loss"
    )
    parser.add_argument(
        '--n_inducing_points',
        type=int,
        default=100,
        help="Number of inducing points to use for variational inference"
    )
    parser.add_argument(
        '--record_interval',
        type=int,
        default=50,
        help="Number of intervals before recording metrics"
    )
    parser.add_argument(
        '--normalize',
        type=int,
        default=0,
        help="Number of inducing points to use for variational inference"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Setting the seed for random sampling"
    )
    parser.add_argument(
        '--time_limit',
        '--output',
        type=str,
        default="200:00",
        help="Limit on training time. Format is [hour, minute]."
    )


    args = parser.parse_args()

    run(args)