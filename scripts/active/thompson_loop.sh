for index in 0 1 2 3 4 5 6 7 8 9 10
do
	for q in 5 10 96
	do
		for acquisition in 'ThompsonSampling' 'ExpectedImprovement' 'ProbabilityOfImprovement' 'UpperConfidenceBound' 'Random' 'Human'
		do
			for num_epochs in 400 600
			do
			 bsub -q gpuqueue -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 1:00 -o %J.stdout -eo %J.stderr python thompson_sampling_plot.py --num_epochs $num_epochs --acquisition $acquisition --q $q --index $index --data moonshot
			done
		done
	done
done