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
        output_regressor_class=getattr(pinot.regressors, args.output_regressor),
    )

    if torch.cuda.is_available():
        net = net.to(torch.device('cuda:0'))

    # get the entire dataset
    ds = getattr(pinot.data, args.data)()

    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds)

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(":")]
    assert len(partition) == 2, "only training and test here."


    ds_tr, ds_te = ds.split(partition)


    if torch.cuda.is_available():
        ds_tr = ds_tr.to(torch.device('cuda:0'))
        ds_te = ds_te.to(torch.device('cuda:0'))

    ds_te = ds_te.view('fixed_size_batch', batch_size=len(ds_te))

    if "Exact" in args.output_regressor:
        ds_tr = ds_tr.view('fixed_size_batch', batch_size=len(ds_tr))
    else:
        ds_tr = ds_tr.view('fixed_size_batch', batch_size=args.batch_size)

    if "Variational" in args.output_regressor:
        optimizer = torch.optim.Adam(
            [
                {
                    'params': net.representation.parameters(),
                    'lr': args.lr,
                },
                {
                    'params': net.output_regressor.kernel.parameters(),
                    'lr': args.lr,
                },
                {
                    'params': [
                        net.output_regressor.log_sigma,
                        net.output_regressor.x_tr,
                        net.output_regressor.y_tr_sigma_tril,
                        net.output_regressor.y_tr_sigma_diag
                        ],
                    'lr': args.lr,
                },
                {
                    'params': [
                        net.output_regressor.y_tr_mu,
                    ],
                    'lr': 1e-1
                }
            ])
    
    else:
        optimizer = pinot.app.utils.optimizer_translation(
            args.optimizer, weight_decay=0.01, lr=args.lr,
        )(net)

    train_and_test = pinot.app.experiment.TrainAndTest(
        net=net,
        data_tr=ds_tr,
        data_te=ds_te,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        record_interval=10,
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
