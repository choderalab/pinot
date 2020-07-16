#BSUB -q cpuqueue
#BSUB -o %J.stdout


bsub -q gpuqueue -J tune -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 4:00 -o %J.stdout -eo %J.stderr python ../../pinot/app/train_old.py --data moonshot_mixed --output_regressor BiophysicalVariationalGaussianProcessRegressor --n_epochs 500 
 
