#BSUB -q cpuqueue
#BSUB -o %J.stdout

for opt in 'Adam'
do
    for layer in 'GraphConv' # 'EdgeConv' 'SGConv' 'GINConv' 'TAGConv' 'SAGEConv'
    do
        for lr in '1e-3'
        do
            for output_regressor in 'NeuralNetworkRegressor'
            do
                
                name="__"$output_regressor
                bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 1:00 -o %J.stdout -eo %J.stderr python ../../pinot/app/train_old.py --layer $layer --optimizer $opt --lr $lr --out $name --n_epochs 50 --config 32 tanh 32 tanh 32 tanh --output_regressor $output_regressor 

done
done
done
done
