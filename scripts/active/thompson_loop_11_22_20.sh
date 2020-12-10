#BSUB -o %J.stdout
for units in 32
do
    for unit_type in "GraphSAGE"
    do
        for n_layers in 6
        do
            layer="$unit_type $units activation tanh "
            architecture=$(for a in `eval echo {1..$n_layers}`; do echo $layer; done)
            for inducing_pt in 20
            do
                for annealing in 0.0 0.2 1.0
                do
                    for regressor in 'vgp'
                    do
                        for sample_frac in 0.1
                        do
							for index in 0 1 2 3 4 5 6 7 8
							do
								for q in 5 10 96
								do
									for acquisition in 'ThompsonSampling' 'ExpectedImprovement' 'ProbabilityOfImprovement' 'UpperConfidenceBound' 'Random'
									do

			                            name="${regressor}_${n_layers}_${unit_type}_${units}_${sample_frac}_${annealing}_${inducing_pt}_${acquisition}_q${q}_${index}"
			                            bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 36:00 -o "logs_$name.stdout" -eo "logs_$name.stderr" \
			                            python3 thompson_sampling_plot.py --data mpro_hts --n_epochs 1000 --cuda --regressor_type $regressor --architecture $architecture \
			                            --acquisition $acquisition --sample_frac $sample_frac --log "${name}.logs" --n_inducing_points $inducing_pt --annealing $annealing --q $q \
			                            --index $index
			                        done
			                    done
			                done
                        done
                    done
                done
            done
        done
    done
done