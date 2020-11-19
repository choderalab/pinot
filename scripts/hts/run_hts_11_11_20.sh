#BSUB -o %J.stdout
for units in 32 64 128 256
do
    for unit_type in "GraphSAGE"
    do
        for n_layers in 3 4 5 6
        do
            layer="$unit_type $units activation tanh "
            architecture=$(for a in `eval echo {1..$n_layers}`; do echo $layer; done)
            for inducing_pt in 20 50 100 200
            do
                for annealing in 0.0 0.2 1.0
                do
                    for regressor in 'vgp'
                    do
                        for sample_frac in 0.1
                        do
                            name="${regressor}_${n_layers}_${unit_type}_${units}_${sample_frac}_${annealing}_${inducing_pt}"
                            bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 36:00 -o "logs_$name.stdout" -eo "logs_$name.stderr" python3 hts_supervised.py --data mpro_hts --n_epochs 1200 --cuda --regressor_type $regressor --architecture $architecture --sample_frac $sample_frac --log "${name}.logs" --record_interval 50 --n_inducing_points $inducing_pt --annealing $annealing
                        done
                    done
                done
            done
        done
    done
done
