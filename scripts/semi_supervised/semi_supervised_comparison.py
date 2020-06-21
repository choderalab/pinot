import argparse

parser = argparse.ArgumentParser("Semi-supervised vs supervised comparison")

parser.add_argument(
    '--regressor_type', 
    type=str,
    default='gp',
    choices=["gp", "nn", "vgp"],
    help="Type of output regressor, Gaussian Process, Variational GP or Neural Networks"
)
parser.add_argument('--layer',
    type=str,
    default='GraphConv',
    help="Type of graph convolution layer"
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
    '--labeled_data',
    type=str,
    default="moonshot",
    help="Labeled data set name"
)

parser.add_argument(
    '--unlabeled_data',
    type=str,
    default="moonshot_unlabeled_small",
    help="Unlabeled data set name"
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
    '--embedding_dim',
    type=int,
    default=64,
    help="Embedding dimension for generative model"
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
    default="semi_learning_curve",
    help="Name of folder to store results"
)

parser.add_argument(
    '--log',
    type=str,
    default="logs",
    help="Log file"
)

args = parser.parse_args()

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

#############################################################################
#
#                   LOAD DATA
#
#############################################################################

start = time.time()
data_labeled_all = getattr(pinot.data, args.labeled_data)()
data_unlabeled_all = getattr(pinot.data, args.unlabeled_data)()

data_labeled_all = [(g.to(device), y.to(device)) for (g,y) in data_labeled_all]

data_unlabeled_all = [(g.to(device), y.to(device)) for (g,y) in data_unlabeled_all]

data_labeled = [(g, y) for (g,y) in data_labeled_all if ~torch.isnan(y)]


# Split the labeled moonshot data into training set and test set
train_labeled, test_labeled = pinot.data.utils.split(data_labeled, [0.8, 0.2])

# Do minibatching on LABELED data
batch_size = 32
one_batch_train_labeled = pinot.data.utils.batch(train_labeled, len(train_labeled))
end = time.time()
logging.debug("Finished loading all data after {} seconds".format(end-start))

#############################################################################
#
#                   TRAIN A SUPERVISED MODEL FIRST
#
#############################################################################


def get_net_and_optimizer(args, unsup_scale):
    layer = pinot.representation.dgl_legacy.gn(model_name=args.layer)

    representation = pinot.representation.sequential.Sequential(
        layer,
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
        weight_decay=0.01,
    )
    return net.to(device), optimizer(net)

def train_and_test_semi_supervised(net, optimizer, semi_data, labeled_train, labeled_test, n_epochs):
    train = pinot.app.experiment.Train(net=net, data=semi_data, optimizer=optimizer, n_epochs=n_epochs)
    train.train()

    train_metrics = pinot.app.experiment.Test(net=net, data=labeled_train, states=train.states, metrics=[pinot.metrics.r2, pinot.metrics.avg_nll, pinot.metrics.rmse])
    train_results = train_metrics.test()

    test_metrics = pinot.app.experiment.Test(net=net, data=labeled_test, states=train.states, metrics=[pinot.metrics.r2, pinot.metrics.avg_nll, pinot.metrics.rmse])
    test_results = test_metrics.test()

    return train_results, test_results

supNet, optimizer = get_net_and_optimizer(args, 0.)

start = time.time()
# Use batch training if exact GP or mini-batch if NN/variational GP
train = one_batch_train_labeled if args.regressor_type == "gp" else pinot.data.utils.batch(train_labeled, batch_size)
suptrain, suptest = train_and_test_semi_supervised(supNet, optimizer, train, train_labeled, test_labeled, args.n_epochs)

end = time.time()
logging.debug("Finished training supervised net after {} seconds".format(end-start))


sup_train_metrics = {}
sup_test_metrics  = {}
for metric in suptrain.keys():
    sup_train_metrics[metric] = suptrain[metric]["final"]
    sup_test_metrics[metric]  = suptest[metric]["final"]

#############################################################################
#
#                   DO LEARNING CURVE EXPERIMENT
#
#############################################################################


vols = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

metrics = list(suptrain.keys())

learning_curve_train = {}
learning_curve_test  = {}
for key in metrics:
    learning_curve_train[key] = []
    learning_curve_test[key]  = []

# Do learning curve experiments
for volume in vols:
    start = time.time()
    # Get a subset of unlabeled data
    train_unlabeled, _ = pinot.data.utils.split(data_unlabeled_all, [volume, 1-volume])

    # Mix the train moonshot labeled with moonshot unlabeled to get semisupervised data
    train_semi = pinot.data.utils.prepare_semi_supervised_data(train_unlabeled, train_labeled)

    # Initialize the model
    # The unsupervised weighting constant is amount of labeled / amount of unlabeled data used for training
    unsup_scale = float(len(train_labeled))/(len(train_unlabeled) + len(train_labeled))

    semiNet, optimizer = get_net_and_optimizer(args, unsup_scale)

    # Train data is either mini-batched (NN / variational GP) or 1 batch (exact GP)
    train = pinot.data.utils.batch(train_semi, len(train_semi)) if args.regressor_type == "gp" else  pinot.data.utils.batch(train_semi, batch_size)

    semitrain, semitest = train_and_test_semi_supervised(semiNet, optimizer, train, train_labeled, test_labeled, args.n_epochs)

    for key in metrics:
        learning_curve_train[key].append(semitrain[key]["final"])
        learning_curve_test[key].append(semitest[key]["final"])
    
    end = time.time()
    logging.debug("Finished training with {} of augmented unlabeled data after {} seconds".format(volume, end-start))


#############################################################################
#
#                   PLOTTING
#
#############################################################################

import matplotlib.pyplot as plt

def plot_learning_curve(learning_curve, vols, sup_metric, savefile=None):
    metrics = list(learning_curve.keys())
    num_metric = len(metrics)

    if num_metric == 1:
        fig1, axs1 = plt.subplots()
        axs1 = [axs1]
    else:
        fig1, axs1 = plt.subplots(num_metric, 1)
        fig1.set_size_inches(15, 12)
        axs1 = axs1.flatten()

    colors = plt.get_cmap('tab10')
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.figure(figsize=(4, 3))
    
    for i, metric in enumerate(metrics):        
        # Plot the metric for semi supervised and then supervised
        metric_track = learning_curve[metric]
        axs1[i].plot(vols, metric_track) # Learning curve
        axs1[i].axhline(sup_metric[metric], 0, 1, color="r") # Just a horizontal line

        axs1[i].set_ylabel(metric)
        axs1[i].set_xlabel("Volume of unlabeled data")
        axs1[i].legend(["Semi-Supervised", "Supervised"])

    if savefile is not None:
        fig1.savefig(savefile)

logging.debug("Finished training all the semi-supervised model, now saving plots")

plot_learning_curve(learning_curve_train, vols, sup_train_metrics, os.path.join(args.output, "{}_{}_{}_{}_train_learning_curve.png".format(args.regressor_type, args.layer, args.architecture, args.n_epochs)))

plot_learning_curve(learning_curve_test, vols, sup_test_metrics, os.path.join(args.output, "{}_{}_{}_{}_test_learning_curve.png".format(args.regressor_type, args.layer, args.architecture, args.n_epochs)))
