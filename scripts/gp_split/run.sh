#BSUB -q cpuqueue
#BSUB -o %J.stdout

for layer in 'GraphConv' 
do
    for idx in {1..99}
    do
        name="_gp_"$idx
        bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 0:30 -o $name/%J.stdout -eo %J.stderr python ../../pinot/app/gp_train.py --layer $layer --out $name --n_epochs 500 --lr 1e-3 --config 32 tanh 32 tanh 32 tanh --lr 1e-3 --partition $idx":"`expr 100 - $idx`

    done
done

