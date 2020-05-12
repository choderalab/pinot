import torch
import pinot

def run():
    def experiment_generating_fn(param_dict):
        param_transform = param_dict['param_transform']

        if 'log_sigma' in param_dict:
            LOG_SIGMA = torch.tensor(param_dict['log_sigma'])
            LOG_SIGMA.requires_grad = True
            param_transform = lambda mu, log_sigma: (mu, LOG_SIGMA)

        net = pinot.Net(
            representation=pinot.representation.Sequential(
                pinot.representation.dgl_legacy.gn(model_name='GraphConv', kwargs={}),
                [32, 'tanh', 32, 'tanh', 32, 'tanh']),
            parameterization=pinot.regression.Linear(32, 2),
            param_transform=param_transform)

        if 'log_sigma' in param_dict:
            optimizer=torch.optim.Adam(
                list(net.parameters()) + [LOG_SIGMA], 1e-3)
        else:
            optimizer=torch.optim.Adam(net.parameters(), 1e-3),

        ds = pinot.data.utils.batch(
            pinot.data.esol(),
            32)

        ds_tr, ds_te = pinot.data.utils.split(
            ds,
            [4, 1])

        train_and_test = pinot.TrainAndTest(
            net=net,
            data_tr=ds_tr,
            data_te=ds_te,
            optimizer=torch.optim.Adam(net.parameters(), 1e-3),
            n_epochs=3000)

        return train_and_test

    multiple_train_and_test = pinot.MultipleTrainAndTest(
        experiment_generating_fn=experiment_generating_fn,
        param_dicts=[
            {
                '#': 'homoscedastic, fixed sigma',
                'param_transform': lambda mu, log_sigma: (mu, torch.tensor(1.0))
            },
            {
                '#': 'homoscedastic, trainable sigma',
                'param_transform': None,
                'log_sigma': 1.0
            },
            {
                '#': 'heteroscedastic, fixed sigma',
                'param_transform': lambda mu, log_sigma: (mu, torch.exp(log_sigma))
            },
        ])

    _ = multiple_train_and_test.run()
    
    html_string = pinot.app.report.html_multiple_train_and_test(multiple_train_and_test.results)
    
    f_handle = open('results.html', 'w')
    f_handle.write(html_string)
    f_handle.close()
    
if __name__ == '__main__':
    run()
