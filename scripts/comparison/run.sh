#BSUB -q cpuqueue
#BSUB -o %J.stdout

for opt in 'Adam'
do
    for layer in 'GraphConv' # 'EdgeConv' 'SGConv' 'GINConv' 'TAGConv' 'SAGEConv'
    do
        for lr in 1e-3 1e-4
        do
            for output_regressor in 'VariationalGaussianProcessRegressor'
            do
                for log_sigma in -3.0
                do
                    for mu_initializer_std in 0.1
                    do
                        for sigma_initializer_value in -2
                        do
                            for n_inducing_points in 50 100 150 200
                            do
                                for n_epochs in 500 1000 1500
                                do

                                    name="_"$output_regressor"_"$lr"_"$log_sigma"_"$mu_initializer_std"_"$sigma_initializer_value"_"$n_inducing_points"_"$n_epochs
                                    bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 1:00 -o %J.stdout -eo %J.stderr python tune_gp.py --layer $layer --optimizer $opt --lr $lr --out $name --n_epochs $n_epochs --config 32 tanh 32 tanh 32 tanh --output_regressor $output_regressor --log_sigma $log_sigma --mu_initializer_std $mu_initializer_std --sigma_initializer_value $sigma_initializer_value --n_inducing_points $n_inducing_points --data moonshot 

done
done
done
done
done
done
done
done
done
