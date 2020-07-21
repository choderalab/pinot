for index in 0 1 2 3 4 5 6 7 8 9 10
do
	for q in 5 10 96
	do
		for acquisition in 'ThompsonSampling' 'ExpectedImprovement' 'ProbabilityOfImprovement' 'UpperConfidenceBound' 'Uncertainty' 'Random' 'Human'
		do
			bsub -q gpuqueue -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 1:00 -o %J.stdout -eo %J.stderr python active_plot.py --acquisition $acquisition --q $q --index $index --data moonshot_sorted --num_epochs 500
		done
	done
done
