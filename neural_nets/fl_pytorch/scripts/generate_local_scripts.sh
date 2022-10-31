#!/usr/bin/env bash

export datasets=("cifar10_fl")
export g_learning_rates=("4.00" "3.75" "3.00" "2.5" "2.00" "1.25" "1.0" "0.75" "0.5" "0.25" "0.2" "0.1" "0.06" "0.03" "0.01" "0.003" "0.001" "0.0006")

export wandb_key="694063ca80a8f491d729ddd4674dcd82e3fdf9c9"
export wandb_project="scripts_no_bn_last_layer_with_imgnet_w_final_with_l2norm_regul"

if [[ ! -f "$1" ]]
then
   echo "File '$1' with command line does not exist. Press any key to continue."
   read
   exit -1
fi

for dataset in "${datasets[@]}"
do
for g_lr in "${g_learning_rates[@]}"
do
   export job_id=$(($RANDOM))

   export dataset
   export g_lr

   fname_with_script=$1
   dest_fname=local_launch_scipt_${dataset}_${g_lr}_${fname_with_script%.*}_${job_id}.sh

   echo "#!/usr/bin/env/bash" > ${dest_fname} 
   echo "" >> ${dest_fname}
   echo "conda activate fl" >> ${dest_fname} 
   echo "" >> ${dest_fname}
   envsubst <${fname_with_script} >>${dest_fname}
done
done

echo "Completed successfully"
