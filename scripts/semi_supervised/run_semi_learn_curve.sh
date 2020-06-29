
logfolder="$PWD/semi-supervised-logs"

for regressor in 'gp' 'vgp' 'nn'
do
    for architecture in 'GraphConv 128 activation tanh GraphConv 128 activation tanh' 'GATConv 64 tanh attention_pool concat GraphConv 128 activation tanh'
    do
        name="${regressor}_${architecture}"
        bsub -q gpuqueue -J gpu-semi -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=16] span[hosts=1]" -W 10:00 -o "${logfolder}/${name}.stdout" -e "${logfolder}/${name}.stderr" /home/wangy1/miniconda3/envs/pinot/bin/python scripts/semi_supervised/semi_supervised_comparison.py --n_epochs 100 --cuda --regressor_type $regressor --architecture $architecture --log "${name}.logs" 
    done
done
