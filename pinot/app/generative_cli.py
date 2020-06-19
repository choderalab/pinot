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
def negative_elbo_loss(net, g, y):
    return net.loss(g).detach()


def run(args):
    if args.info:
        logging.basicConfig(level=logging.INFO)
    logs = logging.getLogger("generative")
    # Load training data
    logs.info("Loading dataset: " + args.background_data)
    background_data = getattr(pinot.data, args.background_data)()
    # Get the number of node features and initialize representation
    # layer as a variational auto-encoder
    input_feat_dim = background_data[0][0].ndata["h"].shape[1]

    train_data, test_data = pinot.data.utils.split(background_data, args.split)
    batched_train_data = pinot.data.utils.batch(train_data, args.batch_size_gen)

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

    logs.info("Doing training...")

    train_and_test = pinot.app.experiment.TrainAndTest(
        net_representation,
        batched_train_data,
        test_data,
        optimizer=gen_optimizer,
        metrics=[negative_elbo_loss],
        n_epochs=args.n_epochs_generative,
    )

    result = train_and_test.run()

    # When finished save the state dictionary
    logs.info(
        "Done training and testing, saving the model state dictionary and results ..."
    )
    os.makedirs(args.out, exist_ok=True)
    torch.save(net_representation, args.out + args.save_model)

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
    parser.add_argument(
        "--layer", type=str, default="GraphConv", help="Type of graph convolution layer"
    )
    parser.add_argument(
        "--hidden_dims_gvae",
        nargs="+",
        type=int,
        default=[128, 128],
        help="Hidden dimensions of the convolution layers",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension (dimension of the encoder's output)",
    )
    parser.add_argument(
        "--background_data",
        type=str,
        default="zinc_tiny",
        help="Background data to pre-train generative model on",
    )
    parser.add_argument(
        "--n_epochs_generative",
        type=int,
        default=50,
        help="Number of epochs of generative model pre-training",
    )
    parser.add_argument(
        "--batch_size_gen",
        type=int,
        default=32,
        help="Batch size for training generative model",
    )
    parser.add_argument(
        "--optimizer_generative",
        type=str,
        default="adam",
        help="Optimizer for generative model pre-training",
    )
    parser.add_argument(
        "--lr_generative",
        type=float,
        default=1e-4,
        help="Learning rate for generative model pre-training",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs="+",
        default=[0.9, 0.1],
        help="Training-testing data split",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gen_result",
        help="Folder to save generative training results to",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="generative_model.pkl",
        help="File to save generative model to",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="INFO mode with more information printing out",
    )
    args = parser.parse_args()
    run(args)
