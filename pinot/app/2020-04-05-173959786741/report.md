2020-04-05-173959786741
===========================
# Model Summary
model=dgl_legacy
config=['256', 'tanh', '256', 'tanh', '256', 'tanh']
distribution=normal
n_params=2
data=esol
batch_size=32
opt=Adam
lr=1e-05
partition=4:1
n_epochs=10
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
        (fc_self): Linear(in_features=128, out_features=256, bias=True)
        (fc_neigh): Linear(in_features=128, out_features=256, bias=True)
      )
    )
    (d2): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=256, out_features=256, bias=True)
        (fc_neigh): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (d4): GN(
      (gn): SAGEConv(
        (feat_drop): Dropout(p=0.0, inplace=False)
        (fc_self): Linear(in_features=256, out_features=256, bias=True)
        (fc_neigh): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (f_out): Linear(in_features=256, out_features=1, bias=True)
  )
  (parameterization): Linear(in_features=256, out_features=2, bias=True)
)
# Time used
24.896366119384766 s
# Performance 
|              |NLL           |rmse          |r2            |
|------------- |------------- |------------- |------------- |
|TRAIN         |2.77          |2.84          |-0.81         |
|TEST          |2.88          |2.92          |-1.03         |
