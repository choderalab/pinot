#BSUB -o %J.stdout
for states in \
'dict_state_reg=vgp_a=2x_128x_GraphSAGE_tanh_n=2200_b=32_wd=0.01_lsp=4_frac=[0.1]_anneal=1.0_induce=40_normalize=0_1_seed=None.p' \
'dict_state_reg=vgp_a=2x_128x_GraphSAGE_tanh_n=2200_b=32_wd=0.01_lsp=4_frac=[0.1]_anneal=0.0_induce=40_normalize=0_1_seed=None.p' \
'dict_state_reg=vgp_a=2x_32x_GraphSAGE_tanh_n=2200_b=32_wd=0.01_lsp=4_frac=[0.1]_anneal=1.0_induce=40_normalize=0_1_seed=None.p' \
'dict_state_reg=vgp_a=2x_32x_GraphSAGE_tanh_n=2200_b=32_wd=0.01_lsp=4_frac=[0.1]_anneal=0.0_induce=40_normalize=0_1_seed=None.p'
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        name="${states}"
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 36:00 -o "logs_$name.stdout" -eo "logs_$name.stderr" \
        python3 hts_debug.py --states $states --cuda --seed $seed \
        --log "${name}.logs" --sample_frac 0.1
    
    done
done