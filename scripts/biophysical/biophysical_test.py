import pinot
from pinot.net import Net
from pinot import data
import torch
import dgl



ds = pinot.data.moonshot_mixed()
# view = ds.view('all_available_pairs', batch_size=32)
view = ds.view('fixed_size_batch', batch_size=32, drop_last=False)
N = ds.number_of_measurements
NG = ds.number_of_unique_graphs

	# import pdb; pdb.set_trace()

# cuda_ds = []
# for d in ds:
# 	di = tuple([i for i in d])
# 	import pdb; pdb.set_trace() 
# 	cuda_ds.append(d)


layer = pinot.representation.dgl_legacy.gn(model_name='GraphConv')

representation = pinot.representation.Sequential(
    layer=layer,
    config=[32, 'tanh', 32, 'tanh', 32, 'tanh']
)


ORC = pinot.regressors.VariationalGaussianProcessRegressor

BRC = pinot.regressors.BiophysicalRegressor

BVGP = pinot.regressors.BiophysicalVariationalGaussianProcessRegressor


#there are two ways to write this model now
#First:

if 0:
	net = pinot.Net(
	    representation=representation,
	    output_regressor_class=ORC,
	    output_likelihood_class=BRC
	)
#and second, which is more intuitive for most of our purposes:
else:
#I recommend using this!!
	net = pinot.Net(
	    representation=representation,
	    output_regressor_class=BVGP,
	    output_likelihood_class=None
	)


import torch
optimizer = torch.optim.Adam(
    net.parameters(),
    1e-3
)

#now we need to loop over the data, no matter which model
for gg, cg, mg in view:
	# bs_ = gg.batch_size
	#return graphs, measurements, concentrations
	loss_data = net.loss(gg, mg, cg)

	print(loss_data.detach().item())

	#if we want to predict we do this:
	distribution_delta_g= net.condition_delta_g(gg,test_ligand_concentration=cg)


