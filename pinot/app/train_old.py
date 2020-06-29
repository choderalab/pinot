# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import pinot
import os
import numpy as np
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):
    """

    Parameters
    ----------
    args :
        

    Returns
    -------

    """
    layer = pinot.representation.dgl_legacy.gn(model_name=args.layer)

    net_representation = pinot.representation.Sequential(
        layer=layer, config=args.config
    )

    net = pinot.Net(
        net_representation,
        output_regressor=getattr(pinot.regressors, args.output_regressor)(32),
    )

    # get the entire dataset
    ds = getattr(pinot.data, args.data)()

    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds)

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(":")]
    assert len(partition) == 2, "only training and test here."

    # batch

    if "Exact" in args.output_regressor:
        ds_tr, ds_te = pinot.data.utils.split(ds, partition)

        ds_tr = pinot.data.utils.batch(ds_tr, len(ds_tr))
        ds_te = pinot.data.utils.batch(ds_te, len(ds_te))

    else:
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
        args.optimizer, weight_decay=0.01, lr=args.lr,
    )(net)

    train_and_test = pinot.app.experiment.TrainAndTest(
        net=net,
        data_tr=ds_tr,
        data_te=ds_te,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
    )

    result = train_and_test.run()

    os.mkdir(args.out)

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
    parser.add_argument("--layer", default="GraphConv", type=str)
    parser.add_argument(
        "--noise_model", default="normal-heteroschedastic", type=str
    )
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument(
        "--config", nargs="*", default=[32, "tanh", 32, "tanh", 32, "tanh"]
    )
    parser.add_argument("--out", default="result", type=str)
    parser.add_argument("--data", default="esol")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--opt", default="Adam")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--partition", default="4:1", type=str)
    parser.add_argument("--n_epochs", default=5)
    parser.add_argument(
        "--output_regressor",
        default="VariationalGaussianProcessOutputRegressor",
        type=str,
    )
    parser.add_argument("--n_inducing_points", default=100, type=int)
    args = parser.parse_args()
    run(args)
