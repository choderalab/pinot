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
        
        opt, lr = param_dict['opt'].split('_')
        lr = float(lr)
        net = param_dict['net']
        
        if net == 'SAGEConv':
            kwargs = {'aggregator_type': 'mean'}

        else:
            kwargs = {}

        
        opt = pinot.app.report.optimizer_translation(
                opt,
                lr,
                kl_loss_scaling=float(1.0 / len(ds_tr)))

        layer = pinot.representation.dgl_legacy.gn(net, kwargs)
        
        net = pinot.Net(
                representation=pinot.representation.Sequential(
                    layer=layer,
                    config=[32, 'tanh', 32, 'tanh', 32, 'tanh']),
                noise_model='normal-homoschedastic-fixed')

        train_and_test = pinot.TrainAndTest(
            net=net,
            data_tr=ds_tr,
            data_te=ds_te,
            optimizer=opt(net),
            n_epochs=1000,
            record_interval=1)

        return train_and_test


    opts = ['Adam', 'BBB', 'AdLaLa']
    nets = ['SAGEConv', 'GraphConv', 'EdgeConv']
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]


    param_dicts = [{
        'net': net,
        'opt': '_'.join([opt, str(lr)]),
        '#': '_'.join([opt, str(lr), net])} for net in nets for opt in opts for lr in lrs]

    multiple_train_and_test = pinot.MultipleTrainAndTest(
        experiment_generating_fn=experiment_generating_fn,
        param_dicts=param_dicts)

    multiple_train_and_test.run()
    
    
    html_string = pinot.app.report.html_multiple_train_and_test_2d_grid(multiple_train_and_test.results)
    f_handle = open('_results_grid_homoschedastic.html', 'w')
    f_handle.write(html_string)
    f_handle.close()

    fig = pinot.app.report.visual_multiple(multiple_train_and_test.results)
    fig.savefig('_results_grid_homoschedastic.png')

 
    torch.save(multiple_train_and_test.results, 'results_grid_homoschedastic.th')

if __name__ == '__main__':
    run()
