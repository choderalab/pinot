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
        os.mkdir(args.output)
        os.mkdir('bin')
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
    architecture_str = f'{n_layers}x_{n_units}x_{layer_type}_{activation}'

    seed = args.seed if args.fix_seed else None
    savefile = (f'reg={args.regressor_type}_a={n_layers}x_{n_units}x'
                f'_{layer_type}_{activation}_n={args.n_epochs}_b={args.batch_size}'
                f'_wd={args.weight_decay}_lsp={args.label_split[0]}_frac={args.sample_frac}'
                f'_anneal={args.annealing}_induce={args.n_inducing_points}_normalize={args.normalize}'
                f'_{args.index}_seed={seed}_pretrainepoch={args.pretrain_epoch}_pretrainfrac={args.pretrain_frac}')

    print(savefile)
    logging.debug("savefile = {}".format(savefile))

    #############################################################################
    #
    #                   LOAD DATA
    #
    #############################################################################

    start = time.time()

    # otherwise, load from scratch
    data = getattr(pinot.data, args.data)(seed=seed)

    # move to cuda
    data = data.to(device)

    # filter out huge outliers
    if args.filter_outliers:
        outlier_threshold = -2
        data.ds = list(filter(lambda x: x[1] > outlier_threshold, data))
    
    # Split the labeled moonshot data into training set and test set
    train_data, test_data = data.split(args.label_split, seed=seed)

    # Set batch size and log
    batch_size = args.batch_size if args.regressor_type != 'gp' else len(train_data)
    end = time.time()
    logging.debug("Finished loading all data after {} seconds".format(end-start))


    #############################################################################
    #
    #                   TRAIN A SUPERVISED MODEL FIRST
    #
    #############################################################################


    def get_net_and_optimizer(args, architecture_str):
        """

        """
        def _get_pretrain_path(args, architecture_str):
            pretrain_dir = '/data/chodera/retchinm/hts_2_21_2021/'
            prefix = f'dict_state_reg=vgp_a={architecture_str}_n=350_b=32_wd=0.01_lsp=4_frac=['
            postfix = ']_anneal=1.0_induce=80_normalize=0_1_seed=0.p'
            pretrain_path = pretrain_dir + prefix + str(args.pretrain_frac) + postfix
            return pretrain_path
        
        representation = pinot.representation.sequential.SequentialMix(
            args.architecture,
        )
        
        if args.regressor_type == "gp":
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
        elif args.regressor_type == "nn":
            output_regressor = pinot.regressors.NeuralNetworkRegressor 
        else:
            output_regressor = pinot.regressors.VariationalGaussianProcessRegressor

        # First train a fully supervised Net to use as Baseline
        net = pinot.Net(
            representation=representation,
            output_regressor_class=output_regressor,
            n_inducing_points=args.n_inducing_points
        )

        optimizer_init = pinot.app.utils.optimizer_translation(
            opt_string=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        net.to(device)

        if args.pretrain_epoch != -1:
            pretrain_path = _get_pretrain_path(args, architecture_str)
            states = pickle.load(open(pretrain_path, 'rb'))
            states_idx = states[args.pretrain_epoch]
            states_idx_representation = {
                k: v for k, v in states_idx.items()
                if 'representation' in k
            }
            net.representation.load_state_dict(states_idx_representation)
            optimizer = optimizer_init(net.output_regressor)
        else:
            optimizer = optimizer_init(net)
        
        return net, optimizer


    supNet, optimizer = get_net_and_optimizer(args, architecture_str)

    start = time.time()
    
    # mini-batch if we're using variational GP
    train_data = train_data.batch(batch_size)

    # get results
    results = pinot.app.experiment.train_and_test(
        supNet,
        train_data,
        test_data,
        optimizer,
        n_epochs=args.n_epochs,
        record_interval=args.record_interval,
        annealing=args.annealing,
        state_save_file=savefile,
        time_limit=args.time_limit,
        out_dir=args.output
    )

    end = time.time()
    
    # logging and clean-up
    logging.debug("Finished training supervised net after {} seconds and save state dict".format(end-start))
    torch.save(supNet.state_dict(), os.path.join(args.output, savefile + "_sup.th"))

    pickle.dump(results['train'], open(f'{args.output}/train_results_{savefile}.p', 'wb'))
    pickle.dump(results['test'], open(f'{args.output}/test_results_{savefile}.p', 'wb'))


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser("pIC50 supervised learning")

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
        default="moonshot_pic50",
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
        '--time_limit',
        type=str,
        default="200:00",
        help="Limit on training time. Format is hour:minute."
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
        '--fix_seed',
        action="store_true",
        default=False,
        help="Whether to fix random seed"
    )
    parser.add_argument(
        '--filter_outliers',
        action="store_true",
        default=False,
        help="Whether to filter huge outliers."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Setting the seed for random sampling"
    )
    parser.add_argument(
        '--pretrain_frac',
        type=float,
        default=0.1,
        help="Which of pretrained models to use by fraction of dataset"
    )
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default="/data/chodera/retchinm/hts_2_21_2021/",
        help="Path where the pretrained model states are stored"
    )
    parser.add_argument(
        '--pretrain_epoch',
        type=int,
        default=-1,
        help="Epoch of training curve for pretrained representation; -1 means no pretraining"
    )


    args = parser.parse_args()

    run(args)
