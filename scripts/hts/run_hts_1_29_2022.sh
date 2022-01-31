#BSUB -o %J.stdout
for units in 32
do

for unit_type in "GraphSAGE"
do

for n_layers in 3
do

layer="$unit_type $units activation relu "
architecture=$(for a in `eval echo {1..$n_layers}`; do echo $layer; done)

for inducing_pt in 100
do

for annealing in 1.0
do

for regressor in 'vgp'
do

for sample_frac in 0.1 1.0
do

for mu_mean in -0.5 0.5
do

for mu_std in 0.1 0.4
do

for std_value in -3.0 -1.0
do

for normalize in 0
do

for seed in 0
do

for filter_threshold in -2.0
do

for filter_neg_train in 0 1
do

for filter_neg_test in 0 1
do

name="${regressor}_${n_layers}_${unit_type}_${units}_${sample_frac}_${annealing}_${inducing_pt}_${normalize}_${seed}_${mu_mean}_${mu_std}_${std_value}"
                bsub \
-q gpuqueue \
-n 2 \
-gpu "num=1:j_exclusive=yes" \
-R "rusage[mem=12] span[hosts=1]" \
-W 65:00 \
-o "logs_$name.stdout" \
-eo "logs_$name.stderr" \
    python3 hts_supervised_functional.py \
--data mpro_hts \
--n_epochs 350 \
--cuda \
--regressor_type $regressor \
--architecture $architecture \
    --sample_frac $sample_frac \
--log "${name}.logs" \
--record_interval 50 \
--n_inducing_points $inducing_pt \
--annealing $annealing \
    --normalize $normalize \
--time_limit "60:00" \
--filter_outliers \
--fix_seed \
--seed $seed \
--output "/data/chodera/retchinm/hts_4_11_2021" \
--mu_mean $mu_mean \
--mu_std $mu_std \
--std_value $std_value \
--filter_threshold $filter_threshold \
--filter_neg_train $filter_neg_train \
--filter_neg_test $filter_neg_test
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
done
done
done
done
done