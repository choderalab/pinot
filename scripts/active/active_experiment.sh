
output_folder="aug-13"
error_log="$PWD/$output_folder/active-sup" # <----------------------------- What folder
output_log="$PWD/$output_folder/active-sup"
pinot_python="/home/nguyenm5/anaconda3/envs/pinot/bin/python"
script="scripts/active/belief_active_plot.py"   # active_plot.py"  #<----------------------------- what PYTHON script

net='semi_vgp' #'ExactGaussianProcessRegressor' #  <---------------------------- Semi supervised? Note the architecture also

envvar="PYTHONPATH=$PWD:$PYTHONPATH"
Jname="active-init"   # <---------------------- Task name

data="moonshot_sorted" # <--------------------------- which data set

num_trials=1
# Do we have Background Data? Note also that the semi-supervised model can also handle no background data
unlabeled_data="--unlabeled_data moonshot_unlabeled_small" # "--unlabeled_data moonshot_unlabeled_all"  # <---------------- Do we have background data???
q=96
device="cuda:0"
num_inducing_points=0
num_inner_opt_rounds=0

for net in 'semi_vgp' 'semi_gp'
do
    for volume in 0.01 #0.4 1.0
    do

    #for i in 1.0,50,80,200 1.0,20,80,200 0.4,50,200,100 0.4,20,200,100; do 
    #IFS=',' read volume num_inducing_points num_epochs num_inner_opt_rounds <<< "${i}"
    
    for acquisition in 'ThompsonSampling' # 'UpperConfidenceBound'
    do

    for index in 1
    do

    for seed in 2666 #417 8888 11 1637
    do
         num_epochs=1
         round=6  #$((576 / $q))
         name=${acquisition}'_'${num_epochs}
         bsub -sp 25 -q gpuqueue -J $Jname -env "$envvar" -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 12 -gpu \
         "num=1:j_exclusive=yes" -R "rusage[mem=16] span[hosts=1]" -W 120:00 -o "${error_log}/${name}.stdout" -e \
         "${error_log}/${name}.stderr" $pinot_python $script --net $net --num_rounds $round --num_trials $num_trials \
         --num_epochs $num_epochs --data $data --q $q --acquisition $acquisition --output_folder $output_log --index \
         $index $unlabeled_data --device $device --seed $seed --unlabeled_volume $volume \
         --num_inducing_points $num_inducing_points --num_inner_opt_rounds $num_inner_opt_rounds
    done
    done
    done
    done
done
