import torch
import pinot

def run():
     
    ds = pinot.data.utils.batch(
        pinot.data.esol(),
        32)

    ds_tr, ds_te = pinot.data.utils.split(
        ds,
        [4, 1])

   
    def experiment_generating_fn(param_dict):
        
        opt = param_dict['opt']
        net = param_dict['net']

        opt = {
            'Adam': lambda net: torch.optim.Adam(net.parameters(), 1e-4),
            'BBB': lambda net: pinot.BBB(torch.optim.Adam(net.parameters(), 1e-4), 0.01),
            'AdLaLa': lambda net: pinot.AdLaLa(
                [
                    {'params': net.representation.parameters(),
                        'h': torch.tensor(1e-5)},
                    {'params': net._output_regression.parameters(),
                        'h': torch.tensor(1e-5)}
                ])

        }[opt]

        
        if net == 'SAGEConv':
            kwargs = {'aggregator_type': 'mean'}

        else:
            kwargs = {}

        layer = pinot.representation.dgl_legacy.gn(net, kwargs)
        
        net = pinot.Net(
                representation=pinot.representation.Sequential(
                    layer=layer,
                    config=[32, 'tanh', 32, 'tanh', 32, 'tanh']))

        train_and_test = pinot.TrainAndTest(
            net=net,
            data_tr=ds_tr,
            data_te=ds_te,
            optimizer=opt(net),
            n_epochs=300,
            record_interval=1)

        return train_and_test


    opts = ['Adam', 'BBB', 'AdLaLa']
    nets = ['SAGEConv', 'GraphConv', 'EdgeConv']

    param_dicts = [{'opt': opt, 'net': net} for opt in opts for net in nets]

    multiple_train_and_test = pinot.MultipleTrainAndTest(
        experiment_generating_fn=experiment_generating_fn,
        param_dicts=param_dicts)

    multiple_train_and_test.run()
    
    html_string = pinot.app.report.html_multiple_train_and_test(multiple_train_and_test.results)
    
    f_handle = open('results_grid.html', 'w')
    f_handle.write(html_string)
    f_handle.close()

if __name__ == '__main__':
    run()
