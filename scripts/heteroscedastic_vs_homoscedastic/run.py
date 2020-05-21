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
        layer = pinot.representation.dgl_legacy.gn()
        
        net = pinot.Net(
                representation=pinot.representation.Sequential(
                    layer=layer,
                    config=[32, 'tanh', 32, 'tanh', 32, 'tanh']),
                noise_model=param_dict['noise_model'])

        train_and_test = pinot.TrainAndTest(
            net=net,
            data_tr=ds_tr,
            data_te=ds_te,
            optimizer=torch.optim.Adam(net.parameters(), 1e-3),
            n_epochs=300,
            record_interval=1)

        return train_and_test

    multiple_train_and_test = pinot.MultipleTrainAndTest(
        experiment_generating_fn=experiment_generating_fn,
        param_dicts=[
            {'noise_model': 'normal-heteroschedastic'},
            {'noise_model': 'normal-homoschedastic'},
            {'noise_model': 'normal-homoschedastic-fixed'}
        ])

    multiple_train_and_test.run()
    
    html_string = pinot.app.report.html_multiple_train_and_test(multiple_train_and_test.results)
    
    f_handle = open('results.html', 'w')
    f_handle.write(html_string)
    f_handle.close()

if __name__ == '__main__':
    run()
