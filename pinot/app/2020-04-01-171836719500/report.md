2020-04-01-171836719500
===========================
# Model Summary
model=dgl_legacy
config=[128, 0.1, 'tanh', 128, 0.1, 'tanh', 128, 0.1, 'tanh']
distribution=normal
n_params=2
data=esol
batch_size=32
opt=Adam
lr=1e-05
partition=4:1
n_epochs=10
report=True
representation_parameter=semi-supervised.ds
regression_parameter=
Net(
  (representation): Sequential(
    (f_in): Sequential(
      (0): Linear(in_features=117, out_features=128, bias=True)
      (1): Tanh()
    )
    (d0): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=128, out_features=128, bias=True)
        (fc_neigh): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    (d3): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=128, out_features=128, bias=True)
        (fc_neigh): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    (d6): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=128, out_features=128, bias=True)
        (fc_neigh): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    (f_out): Linear(in_features=128, out_features=1, bias=True)
  )
  (parameterization): Linear(in_features=128, out_features=2, bias=True)
)
# Time used
11.574890375137329 s
# Performance 
|              |NLL           |
|------------- |------------- |
|TRAIN         |10.61         |
|TEST          |12.66         |
