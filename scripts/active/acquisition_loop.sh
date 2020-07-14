for index in 0 1 2 3 4 5 6 7 8 9 10
do
	for q in 5 10 96
	do
		for acquisition in 'ThompsonSampling' 'WeightedSamplingExpectedImprovement' 'WeightedSamplingProbabilityOfImprovement' 'WeightedSamplingUpperConfidenceBound' 'GreedyExpectedImprovement' 'GreedyProbabilityOfImprovement' 'GreedyUpperConfidenceBound' 'BatchRandom' 'BatchTemporal'
		do
			python plotting_active.py --acquisition $acquisition --q 10 --index $index
		done
	done
done