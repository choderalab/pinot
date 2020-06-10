#BSUB -q cpuqueue
#BSUB -o %J.stdout

for layer in 'GraphConv' 'EdgeConv' 'SAGEConv' 'GINConv' 'SGConv' 'TAGConv' 
do
        name="_gp_"$layer
        bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 0:30 -o $name/%J.stdout -eo %J.stderr python ../../pinot/app/gp_train.py --layer $layer --out $name --n_epochs 100 --lr 1e-3 --config 8 tanh 8 tanh 8 tanh

done

