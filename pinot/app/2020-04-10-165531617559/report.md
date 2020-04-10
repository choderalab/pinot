2020-04-10-165531617559
===========================
# Model Summary
model=dgl_legacy
config=['128', 'tanh', '128', 'tanh', '128', 'tanh']
distribution=normal
n_params=2
data=esol
batch_size=32
opt=Adam
lr=1e-05
partition=4:1
n_epochs=50
report=True
representation_parameter=
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
    (d2): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=128, out_features=128, bias=True)
        (fc_neigh): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    (d4): GN(
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
86.46072030067444 s
# Performance 
|              |NLL           |rmse          |r2            |
|------------- |------------- |------------- |------------- |
|TRAIN         |3.38          |5.75          |-6.46         |
|TEST          |3.44          |5.78          |-6.94         |
