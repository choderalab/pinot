def run(args):
    import pinot
    import torch
    import os
    import logging
    import time

    # Specify accelerator (if any)
    device = torch.device("cuda:0" if args.cuda else "cpu:0")

    # If output folder doesn't exist, create a new one
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    logging.basicConfig(filename=os.path.join(args.output, args.log), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.debug(args)

    savefile = "reg={}_a={}_n={}_b={}_wd={}_lsp={}_percent={}".format(args.regressor_type, args.architecture, args.n_epochs, args.batch_size, args.weight_decay, args.label_split, args.percent_data)
    logging.debug("savefile = {}".format(savefile))

    #############################################################################
    #
    #                   LOAD DATA
    #
    #############################################################################

    start = time.time()
    data_all = getattr(pinot.data, args.data)()

    data_all = [(g.to(device), y.to(device)) for (g,y) in data_all]
    data = [(g, y) for (g,y) in data_all if ~torch.isnan(y)]


    # Split the labeled moonshot data into training set and test set
    train_labeled, test_labeled = pinot.data.utils.split(data, args.label_split)

    # Do minibatching on LABELED data
    batch_size = args.batch_size
    one_batch_train_labeled = pinot.data.utils.batch(train_labeled, len(train_labeled))
    end = time.time()
    logging.debug("Finished loading all data after {} seconds".format(end-start))


    #############################################################################
    #
    #                   TRAIN A SUPERVISED MODEL FIRST
    #
    #############################################################################


    def get_net_and_optimizer(args, unsup_scale):
        representation = pinot.representation.sequential.SequentialMix(
            args.architecture,
        )

        if args.regressor_type == "gp":
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
        elif args.regressor_type == "nn":
            output_regressor = pinot.regressors.NeuralNetworkRegressor 
        else:
            output_regressor = pinot.regressors.VariationalGaussianProcessRegressor

        decoder = pinot.generative.DecoderNetwork

        # First train a fully supervised Net to use as Baseline
        net = pinot.generative.SemiSupervisedNet(
            representation=representation,
            output_regressor=output_regressor,
            decoder=decoder,
            unsup_scale=unsup_scale,
            embedding_dim=args.embedding_dim
        )
        optimizer = pinot.app.utils.optimizer_translation(
            opt_string=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        net.to(device)
        return net, optimizer(net)

    def train_and_test_semi_supervised(net, optimizer, semi_data, labeled_train, labeled_test, n_epochs):
        train = pinot.app.experiment.Train(net=net, data=semi_data, optimizer=optimizer, n_epochs=n_epochs)
        train.train()

        train_metrics = pinot.app.experiment.Test(net=net, data=labeled_train, states=train.states, metrics=[pinot.metrics.r2, pinot.metrics.avg_nll, pinot.metrics.rmse])
        train_results = train_metrics.test()

        test_metrics = pinot.app.experiment.Test(net=net, data=labeled_test, states=train.states, metrics=[pinot.metrics.r2, pinot.metrics.avg_nll, pinot.metrics.rmse])
        test_results = test_metrics.test()

        return train_results, test_results, train.states

    supNet, optimizer = get_net_and_optimizer(args, 0.) # Note that with unsup_scale = 0., this becomes a supervised only model

    start = time.time()
    # Use batch training if exact GP or mini-batch if NN/variational GP
    train = one_batch_train_labeled if args.regressor_type == "gp" else pinot.data.utils.batch(train_labeled, batch_size)
    suptrain, suptest, supstates = train_and_test_semi_supervised(supNet, optimizer, train, train_labeled, test_labeled, args.n_epochs)

    end = time.time()
    logging.debug("Finished training supervised net after {} seconds and save state dict".format(end-start))
    torch.save(supNet.state_dict(), os.path.join(args.output, savefile + "_sup.th"))
    torch.save(supstates, os.path.join(args.output, savefile + "_sup_all_states.th"))

    sup_train_metrics = {}
    sup_test_metrics  = {}
    for metric in suptrain.keys():
        sup_train_metrics[metric] = suptrain[metric]["final"]
        sup_test_metrics[metric]  = suptest[metric]["final"]

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
        default="moonshot",
        help="Labeled data set name"
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help="number of training epochs"
    )
    parser.add_argument(
        '--architecture',
        nargs="+",
        type=str,
        default=[128, "tanh", 128, "tanh"],
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
        default="learning_curve",
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
        '--percent_data',
        nargs="+",
        type=float,
        default=0.1,
        help="Proportion of dataset to use"
    )

    parser.add_argument(
        '--label_split',
        nargs="+",
        type=float,
        default=[0.8,0.2],
        help="Training-testing split for labeled data"
    )

    args = parser.parse_args()

    run(args)