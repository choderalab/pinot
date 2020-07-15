for index in 0 1 2 3 4 5 6 7 8 9 10
do
	for q in 5 10 96
	do
		for acquisition in 'ThompsonSampling' 'ExpectedImprovement' 'ProbabilityOfImprovement' 'UpperConfidenceBound' 'Uncertainty' 'Random' 'Human'
		do
			python active_plot.py --acquisition $acquisition --q $q --index $index
		done
	done
done