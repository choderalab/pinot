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
    layer = pinot.representation.dgl_legacy.gn(model_name=args.layer)

    net_representation = pinot.representation.Sequential(
        layer=layer, config=args.config
    )

    hidden_dim = [
        layer
        for layer in net_representation.modules()
        if hasattr(layer, "out_features")
    ][-1].out_features

    kernel = pinot.inference.gp.kernels.deep_kernel.DeepKernel(
        representation=net_representation,
        base_kernel=pinot.inference.gp.kernels.rbf.RBF(
            scale=torch.zeros(hidden_dim)
        ),
    )

    gpr = pinot.inference.gp.gpr.exact_gpr.ExactGPR(
        kernel, log_sigma=float(args.log_sigma)
    )

    # get the entire dataset
    ds = getattr(pinot.data, args.data)()

    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds)

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(":")]
    assert len(partition) == 2, "only training and test here."

    ds_tr, ds_te = pinot.data.utils.split(ds, partition)

    ds_tr = pinot.data.utils.batch(ds_tr, len(ds_tr))
    ds_te = pinot.data.utils.batch(ds_te, len(ds_te))

    if torch.cuda.is_available():
        ds_tr = [
            (g.to(torch.device("cuda:0")), y.to(torch.device("cuda:0")))
            for g, y in ds_tr
        ]
        ds_te = [
            (g.to(torch.device("cuda:0")), y.to(torch.device("cuda:0")))
            for g, y in ds_te
        ]

        gpr = gpr.to(torch.device("cuda:0"))

    optimizer = pinot.app.utils.optimizer_translation(
        args.optimizer,
        lr=args.lr,
        weight_decay=0.01,
        kl_loss_scaling=1.0 / float(len(ds_tr)),
    )(gpr)

    train_and_test = pinot.app.experiment.TrainAndTest(
        net=gpr,
        data_tr=ds_tr,
        data_te=ds_te,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
    )

    result = train_and_test.run()

    os.mkdir(args.out)

    torch.save(gpr.state_dict(), args.out + "/model_state_dict.th")

    # torch.save(gpr, args.out + '/model.th')

    torch.save(gpr.condition(ds_tr[0][0]), args.out + "/distribution_tr.th")
    torch.save(gpr.condition(ds_te[0][0]), args.out + "/distribution_te.th")

    np.save(
        arr=ds_tr[0][1].detach().cpu().numpy(), file=args.out + "/y_tr.npy"
    )
    np.save(
        arr=ds_te[0][1].detach().cpu().numpy(), file=args.out + "/y_te.npy"
    )

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
    parser.add_argument("--log_sigma", default=-3)

    args = parser.parse_args()
    run(args)
