# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import pinot
import os
import numpy as np
import torch
from pinot.generative.torch_gvae import GCNModelVAE
import logging

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):
    if args.info:
        logging.basicConfig(level=logging.INFO)

    logs = logging.getLogger("pinot")
    net_representation = None
    # If there are no pretrained generative model specified
    if args.pretrained_gen_model is None:
        logs.info(
            "No pretrained model is specified, training generative model"
            + "using background data ..."
        )
        # Load the background training data
        logs.info("Loading dataset: " + args.background_data)
        background_data = getattr(pinot.data, args.background_data)()
        # Get the number of node features and initialize representation
        # layer as a variational auto-encoder
        input_feat_dim = background_data[0][0].ndata["h"].shape[1]

        batched_background_data = pinot.data.utils.batch(
            background_data, args.batch_size_gen
        )

        net_representation = GCNModelVAE(
            input_feat_dim,
            gcn_type=args.layer,
            gcn_hidden_dims=args.hidden_dims_gvae,
            embedding_dim=args.embedding_dim,
        )

        # And then train this model
        gen_optimizer = pinot.app.utils.optimizer_translation(
            args.optimizer_generative, lr=args.lr_generative
        )(net_representation)
        logs.info("Training generative model ...")
        generative_train = pinot.app.experiment.Train(
            net_representation,
            batched_background_data,
            gen_optimizer,
            args.n_epochs_generative,
        )

        generative_train.train()
        # When done, save the generative model
        torch.save(net_representation, args.save_model)
        logs.info(
            "Finished training generative model and saving trained model"
        )

    else:
        # Load the pretrained generative model
        logs.info("Loading pretrained generative model")
        net_representation = torch.load(args.pretrained_gen_model)
        logs.info("Finished loading!")

    # Freeze the gradient if the user does not specify --free_gradient
    if not args.free_gradient:
        for param in net_representation.parameters():
            param.requires_grad = False

    # get the foreground data
    ds = getattr(pinot.data, args.data)()

    # Initialize the Net from with the generative model
    net = pinot.Net(net_representation, noise_model=args.noise_model)

    optimizer = pinot.app.utils.optimizer_translation(
        args.optimizer, lr=args.lr
    )(net)

    # get the entire dataset
    ds = getattr(pinot.data, args.data)()

    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds)

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(":")]
    assert len(partition) == 2, "only training and test here."

    # batch
    ds = pinot.data.utils.batch(ds, batch_size)
    ds_tr, ds_te = pinot.data.utils.split(ds, partition)

    if torch.cuda.is_available():
        ds_tr = [
            (g.to(torch.device("cuda:0")), y.to(torch.device("cuda:0")))
            for g, y in ds_tr
        ]
        ds_te = [
            (g.to(torch.device("cuda:0")), y.to(torch.device("cuda:0")))
            for g, y in ds_te
        ]

        net = net.to(torch.device("cuda:0"))

    optimizer = pinot.app.utils.optimizer_translation(
        args.optimizer, lr=args.lr, kl_loss_scaling=1.0 / float(len(ds_tr))
    )(net)

    train_and_test = pinot.app.experiment.TrainAndTest(
        net=net,
        data_tr=ds_tr,
        data_te=ds_te,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
    )

    result = train_and_test.run()

    os.makedirs(args.out, exist_ok=True)

    torch.save(net.state_dict(), args.out + "/model_state_dict.th")

    with open(args.out + "/architecture.txt", "w") as f_handle:
        f_handle.write(str(train_and_test))

    with open(args.out + "/result_table.md", "w") as f_handle:
        f_handle.write(pinot.app.report.markdown(result))

    curves = pinot.app.report.curve(result)

    for spec, curve in curves.items():
        np.save(args.out + "/" + "_".join(spec) + ".npy", curve)

    with open(args.out + "/result.html", "w") as f_handle:
        f_handle.write(pinot.app.report.html(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # With pretrained generative model
    pretrained_gen_group = parser.add_argument_group(
        "With pretrained generative model:"
    )
    pretrained_gen_group.add_argument(
        "--pretrained_gen_model",
        default=None,
        type=str,
        help="File of pretrained generative model in pkl format",
    )

    # With no pretrained generative model
    group = parser.add_argument_group("With no pretrained generative model")
    group.add_argument(
        "--layer",
        type=str,
        default="GraphConv",
        help="Type of graph convolution layer",
    )
    group.add_argument(
        "--hidden_dims_gvae",
        nargs="+",
        type=int,
        default=[128, 128],
        help="Hidden dimensions of the convolution layers",
    )
    group.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension (dimension of the encoder's output)",
    )
    group.add_argument(
        "--background_data",
        type=str,
        default="zinc_tiny",
        help="Background data to pre-train generative model on",
    )
    group.add_argument(
        "--n_epochs_generative",
        type=int,
        default=50,
        help="Number of epochs of generative model pre-training",
    )
    group.add_argument(
        "--batch_size_gen",
        type=int,
        default=32,
        help="Batch size for training generative model",
    )
    group.add_argument(
        "--optimizer_generative",
        type=str,
        default="adam",
        help="Optimizer for generative model pre-training",
    )
    group.add_argument(
        "--lr_generative",
        type=float,
        default=1e-4,
        help="Learning rate for generative model pre-training",
    )
    group.add_argument(
        "--save_model",
        type=str,
        default="generative_model.pkl",
        help="File to save generative model to",
    )

    # Settings for networks
    net_args = parser.add_argument_group("Net settings")
    net_args.add_argument(
        "--free_gradient",
        action="store_true",
        help="Allow for updating gradients of pretrained generative model",
    )
    net_args.add_argument(
        "--noise_model",
        default="normal-heteroschedastic",
        type=str,
        help="Noise model for predictive distribution",
    )
    net_args.add_argument(
        "--optimizer", default="adam", type=str, help="Choice of ptimizer"
    )
    net_args.add_argument(
        "--out",
        default="result",
        type=str,
        help="Folder to print out results to",
    )
    net_args.add_argument("--data", default="esol", help="Data set name")
    net_args.add_argument(
        "--batch_size", default=32, type=int, help="Batch size"
    )
    net_args.add_argument(
        "--lr", default=1e-5, type=float, help="Learning rate"
    )
    net_args.add_argument(
        "--partition", default="4:1", type=str, help="Training-testing split"
    )
    net_args.add_argument("--n_epochs", default=5, help="Number of epochs")

    parser.add_argument(
        "--info",
        action="store_true",
        help="INFO mode with more information printing out",
    )

    args = parser.parse_args()
    run(args)
