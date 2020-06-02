# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import pinot
import os
import numpy as np
import torch
from pinot.generative.torch_gvae import GCNModelVAE

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):

    # get the entire dataset
    ds= getattr(
        pinot.data,
        args.data)()

    # Get the number of node features and initialize representation
    # layer as a variational auto-encoder
    input_feat_dim = ds[0][0].ndata["h"].shape[1]
    net_representation = GCNModelVAE(input_feat_dim, 
            gcn_type=args.layer,
            gcn_hidden_dims=args.hidden_dims_gvae,
            embedding_dim=args.embedding_dim)

    net = pinot.Net(
        net_representation,
        noise_model=args.noise_model)

    optimizer = pinot.app.utils.optimizer_translation(
        args.optimizer,
        lr=args.lr)(net)


    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds)

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(':')]
    assert len(partition) == 2, 'only training and test here.'

    # batch
    ds = pinot.data.utils.batch(ds, batch_size)
    ds_tr, ds_te = pinot.data.utils.split(ds, partition)

    train_and_test = pinot.app.experiment.TrainAndTest(
        net=net,
        data_tr=ds_tr,
        data_te=ds_te,
        optimizer=optimizer,
        n_epochs=args.n_epochs)

    result = train_and_test.run()

    os.mkdir(args.out)

    torch.save(net.state_dict(), args.out + '/model_state_dict.th')

    with open(args.out + '/architecture.txt', 'w') as f_handle:
        f_handle.write(str(train_and_test))

    with open(args.out + '/result_table.md', 'w') as f_handle:
        f_handle.write(
            pinot.app.report.markdown(result))

    curves = pinot.app.report.curve(result)

    for spec, curve in curves.items():
        np.save(
            args.out + '/' + '_'.join(spec) + '.npy',
            curve)

    with open(args.out + '/result.html', 'w') as f_handle:
        f_handle.write(pinot.app.report.html(
            result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='GraphConv', type=str, help="Type of graph convolution layer")
    parser.add_argument('--hidden_dims_gvae',nargs='+', type=int, default=[128, 128], help="Hidden dimensions of the convolution layers")
    parser.add_argument('--embedding_dim', type=int, default=64, help="embedding dimension (dimension of the encoder's output)")
    parser.add_argument('--noise_model', default='normal-heteroschedastic', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--out', default='result', type=str, help="Folder to print out results to")
    parser.add_argument('--data', default='esol', help="Data set name")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
    parser.add_argument('--partition', default='4:1', type=str, help="Training-testing split")
    parser.add_argument('--n_epochs', default=5, help="Number of epochs")

    args = parser.parse_args()
    run(args)
