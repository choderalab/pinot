#BSUB -q cpuqueue
#BSUB -o %J.stdout

for opt in 'Adam'
do
    for layer in 'GraphConv' # 'EdgeConv' 'SGConv' 'GINConv' 'TAGConv' 'SAGEConv'
    do
        for lr in 1e-3 1e-4 1e-5
        do
            for output_regressor in 'VariationalGaussianProcessRegressor'
            do
                for log_sigma in 1.0 0.0 -1.0 -2.0 -3.0
                do
                    for mu_initializer_std in 0.1 0.01 0.001
                    do
                        for sigma_initializer_std in 0.1 0.001 0.0001
                        do

                name="_"$output_regressor"_"$lr"_"$log_sigma"_"$mu_initializer_std"_"$sigma_initializer_std
                bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 1:00 -o %J.stdout -eo %J.stderr python tune_gp.py --layer $layer --optimizer $opt --lr $lr --out $name --n_epochs 500 --config 32 tanh 32 tanh 32 tanh --output_regressor $output_regressor --log_sigma $log_sigma --mu_initializer_std $mu_initializer_std --sigma_initializer_std $sigam_initializer_std 

done
done
done
done
done
done
done
