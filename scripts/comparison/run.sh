#BSUB -q cpuqueue
#BSUB -o %J.stdout

for opt in 'adam' 'bbb' 'sgld'
do
    for layer in 'SAGEConv' 'GraphConv' 'EdgeConv'
    do
        for lr in '1e-2' '1e-3' '1e-4'
        do
            name="_"$opt"_"$layer"_"$lr
            bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 8:00 -o $name/%J.stdout -eo %J.stderr python ../../pinot/app/supervised_train.py --layer $layer --optimizer $opt --lr $lr --out $name --n_epochs 100

done
done
done

