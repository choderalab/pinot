for index in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
	for q in 5 10 96
	do
		for acquisition in 'ThompsonSampling' 'ExpectedImprovement' 'ProbabilityOfImprovement' 'UpperConfidenceBound' 'Random' 'Human'
		do
			for num_epochs in 400 800 1200
			do
				python thompson_sampling_plot.py --num_epochs $num_epochs --acquisition $acquisition --q $q --index $index --data moonshot
			done
		done
	done
done