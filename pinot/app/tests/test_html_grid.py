import pytest
import torch
import pinot

def test_html_grid():

    ds = pinot.data.utils.batch(
        pinot.data.esol()[:10],
        5)

    ds_tr, ds_te = pinot.data.utils.split(
        ds,
        [1, 1])
    
    def experiment_generating_fn(param_dict):


        net = pinot.Net(
            representation=pinot.representation.Sequential(
                pinot.representation.dgl_legacy.GN,
                [32, 'tanh', 32, 'tanh', 32, 'tanh']))

        optimizer=torch.optim.Adam(net.parameters(), param_dict['lr'])

        train_and_test = pinot.TrainAndTest(
            net=net,
            data_tr=ds_tr,
            data_te=ds_te,
            optimizer=optimizer,
            n_epochs=1,
            record_interval=1)

        return train_and_test

    
    param_dicts = [{'lr': lr, 'model': model} for lr in [1e-2, 1e-3] for model in [
        'EdgeConv', 'GraphConv']]

    multiple_train_and_test = pinot.MultipleTrainAndTest(
        experiment_generating_fn=experiment_generating_fn,
        param_dicts=param_dicts)

    multiple_train_and_test.run()
    
    html_string = pinot.app.report.html_multiple_train_and_test(multiple_train_and_test.results)
    
    f_handle = open('results.html', 'w')
    f_handle.write(html_string)
    f_handle.close()



