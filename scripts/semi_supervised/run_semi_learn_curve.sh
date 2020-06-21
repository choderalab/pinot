#BSUB -q cpuqueue
#BSUB -o %J.stdout


for layer in 'GraphConv'
do
    for architecture in '128 tanh 128 tanh' '64 tanh 64 tanh' '64 tanh 64 tanh 64 tanh'
    do
        name="_"$opt"_"$layer"_"$lr
        bsub -q gpuqueue -J $name -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 0:30 -o $name/%J.stdout -eo %J.stderr python pinot/scripts/semi_supervised/semi_supervised_comparison.py --layer $layer --optimizer $opt --architecture $architecture --n_epochs 500 --cuda

    done
done
