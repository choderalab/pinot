#BSUB -o %J.stdout
for units in 32
do
    for unit_type in "GraphSAGE" "GraphConv"
    do
        for n_layers in 3
        do
            layer="$unit_type $units activation relu "
            architecture=$(for a in `eval echo {1..$n_layers}`; do echo $layer; done)
            for inducing_pt in 40 80
            do
                for annealing in 0.0 0.2 1.0
                do
                    for regressor in 'vgp'
                    do
			for normalize in 0 1
			do
			    for seed in 0
			    do
			        for pretrain_epoch in -1 100 250 349
				do
				    for pretrain_frac in 0.4
				    do
                            		    name="${regressor}_${n_layers}_${unit_type}_${units}_${sample_frac}_${annealing}_${inducing_pt}_${normalize}_${seed}"
                            		    bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 3:00 -o "logs_$name.stdout" -eo "logs_$name.stderr" \
			    		    python3 ic50_supervised.py --data moonshot_pic50 --n_epochs 1000 --cuda --regressor_type $regressor --architecture $architecture \
				            --log "${name}.logs" --record_interval 50 --n_inducing_points $inducing_pt --annealing $annealing \
			    		    --normalize $normalize --time_limit "2:30" --filter_outliers --fix_seed --seed $seed --output "/data/chodera/retchinm/pIC50/3_28_2021" \
					    --pretrain_epoch $pretrain_epoch --pretrain_frac $pretrain_frac
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
