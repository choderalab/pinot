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
    
    # Split the labeled moonshot data into training set and test set
    train_data, test_data = data.split(args.label_split)

    # Do minibatching on LABELED data
    batch_size = args.batch_size if args.regressor_type != 'gp' else len(train_data)
    end = time.time()
    logging.debug("Finished loading all data after {} seconds".format(end-start))


    #############################################################################
    #
    #                   TRAIN A SUPERVISED MODEL FIRST
    #
    #############################################################################


    def get_net_and_optimizer(args):
        
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
        optimizer = pinot.app.utils.optimizer_translation(
            opt_string=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        net.to(device)
        return net, optimizer(net)

    def train_and_test(
        net,
        optimizer,
        train_data,
        test_data,
        n_epochs,
        annealing,
        record_interval,
        state_save_file):
        
        train_and_test = pinot.TrainAndTest(
            net=net,
            optimizer=optimizer,
            n_epochs=n_epochs,
            data_tr=train_data,
            data_te=test_data,
            annealing=annealing,
            record_interval=record_interval,
            state_save_file=state_save_file
        )

        result = train_and_test.run()

        return result['training'], result['test']

    supNet, optimizer = get_net_and_optimizer(args)

    start = time.time()
    
    # mini-batch because we're using variational GP
    train_data = train_data.batch(batch_size)
    train_results, test_results = train_and_test(
        supNet,
        optimizer,
        train_data,
        test_data,
        args.n_epochs,
        args.annealing,
        args.record_interval,
        savefile
    )

    end = time.time()
    logging.debug("Finished training supervised net after {} seconds and save state dict".format(end-start))
    torch.save(supNet.state_dict(), os.path.join(args.output, savefile + "_sup.th"))

    sup_train_metrics = {}
    sup_test_metrics  = {}
    for metric in train_results.keys():
        sup_train_metrics[metric] = train_results[metric]["final"]
        sup_test_metrics[metric]  = test_results[metric]["final"]

    pickle.dump(train_results, open(f'./{args.output}/train_results_{savefile}.p', 'wb'))
    pickle.dump(test_results, open(f'./{args.output}/test_results_{savefile}.p', 'wb'))

    logging.debug(sup_train_metrics)
    logging.debug(sup_test_metrics)



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