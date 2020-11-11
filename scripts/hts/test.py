def run(args):
    import os
    import logging
    import time
    import pickle

    # If output folder doesn't exist, create a new one
    try:
        os.mkdir('bin')
        os.mkdir(args.output)
    except:
        pass

    logging.basicConfig(filename=os.path.join(args.output, args.log), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.debug(args)

    layer_type = args.architecture[0]
    n_layers = len(args.architecture) // 4
    n_units = args.architecture[1]
    activation = args.architecture[3]

    savefile = f'reg={args.regressor_type}_a={n_layers}x_{n_layers}x_{layer_type}_{activation}_n={args.n_epochs}_b={args.batch_size}_wd={args.weight_decay}_lsp={args.label_split[0]}_frac={args.sample_frac}_anneal={args.annealing}_induce={args.n_inducing_points}_{args.index}'
    print(savefile)
    logging.debug("savefile = {}".format(savefile))

    #############################################################################
    #
    #                   LOAD DATA
    #
    #############################################################################

if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser("HTS supervised learning")

    parser.add_argument(
        '--regressor_type', 
        type=str,
        default='gp',
        choices=["gp", "nn", "vgp"],
        help="Type of output regressor, Gaussian Process, Variational GP or Neural Networks"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help="learning rate of optimizer"
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help="Optimization algorithm"
    )
    parser.add_argument(
        '--data',
        type=str,
        default="mpro_hts",
        help="Labeled data set name"
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=500,
        help="number of training epochs"
    )
    parser.add_argument(
        '--architecture',
        nargs="+",
        type=str,
        default=[32, "tanh", 32, "tanh", 32, "tanh"],
        help="Graph neural network architecture"
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


    args = parser.parse_args()

    run(args)
