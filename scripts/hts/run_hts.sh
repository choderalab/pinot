logfolder="$PWD/hts-logs"

for regressor in 'vgp' 'nn'
do
    for architecture in 'GraphConv 32 activation tanh GraphConv 32 activation tanh GraphConv 32 activation tanh'
    do
    	for sample_frac in 0.01 0.1 0.2 1.0
    	do
	    	name="${regressor}_${architecture}_${sample_frac}"
			bsub -q gpuqueue -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=16] span[hosts=1]" -W 1:00 -o "logs_$name.stdout" -eo "logs_$name.stderr" python3 hts_supervised.py --data mpro_hts --n_epochs 500 --cuda --regressor_type $regressor --architecture $architecture --sample_frac $sample_frac --log "${name}.logs"
		done
	done
done