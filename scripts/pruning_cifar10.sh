#!/bin/bash
echo "Pruning with filter similarity!"

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90
resnet20 54
}


pruning_scratch_resnet110(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet110(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.001 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet56(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet56 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 164 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet56(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet56 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 164 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet32(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  /home/share/data/cifar10 --dataset cifar10 --arch resnet32 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 90 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet32(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet32 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 90 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet20(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  /home/share/data/cifar10 --dataset cifar10 --arch resnet20 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1
}
pruning_scratch_resnet20_distill(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  /home/share/data/cifar10 --dataset cifar10 --arch resnet20 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1 \
--kd \
--T 10 --alpha $5 \
--teacher_model_path ./pretrained/cifar10/resnet20/checkpoint.pth.tar
}
pruning_pretrain_resnet20(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet20 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1
}

jump(){

(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm1_ratedist0.4_varience1 1 0.4)&
(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm1_ratedist0.4_varience2 1 0.4)&
(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm1_ratedist0.4_varience3 1 0.4)&


(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm0.7_ratedist0.1_varience1 0.7 0.1)&
(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm0.7_ratedist0.1_varience2 0.7 0.1)&
(pruning_pretrain_resnet56 0 /data/yahe/cifar_GM/pretrain_0.01/cifar10_resnet56_ratenorm0.7_ratedist0.1_varience3 0.7 0.1)&

}




run32(){

#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm1_ratedist0.4_varience1 1 0.4)&
#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm1_ratedist0.4_varience2 1 0.4)&
#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm1_ratedist0.4_varience3 1 0.4)&
#
#
#pruning_scratch_resnet32 0 log/res32/norm0.9_fpgm_0.3_round1 0.9 0.3
pruning_scratch_resnet32 1 log/res32/norm_no_fixed_0.9_fpgm_fixed_0.3_round1 0.9 0.3
#(pruning_scratch_resnet32 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet32_ratenorm1_ratedist0.4_varience2 1 0.4)&
#(pruning_scratch_resnet32 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet32_ratenorm1_ratedist0.4_varience3 1 0.4)&
#
#
#
#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience1 1 0.1)&
#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience2 1 0.1)&
#(pruning_pretrain_resnet32 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience3 1 0.1)&
#
#
#(pruning_scratch_resnet32 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience1 1 0.1)&
#(pruning_scratch_resnet32 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience2 1 0.1)&
#(pruning_scratch_resnet32 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet32_ratenorm0.7_ratedist0.1_varience3 1 0.1)&

}

run20_1(){
(pruning_pretrain_resnet20 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet20_ratenorm1_ratedist0.4_varience1 1 0.4)&
(pruning_pretrain_resnet20 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet20_ratenorm1_ratedist0.4_varience2 1 0.4)&
(pruning_pretrain_resnet20 0 /data/yahe/cifar_GM2/pretrain_0.01/cifar10_resnet20_ratenorm1_ratedist0.4_varience3 1 0.4)&


(pruning_scratch_resnet20 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet20_ratenorm1_ratedist0.4_varience1 1 0.4)&
(pruning_scratch_resnet20 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet20_ratenorm1_ratedist0.4_varience2 1 0.4)&
(pruning_scratch_resnet20 0 /data/yahe/cifar_GM2/scratch/cifar10_resnet20_ratenorm1_ratedist0.4_varience3 1 0.4)&
}
run20_2(){
#pruning_scratch_resnet20 0 log/res20_norm0.9_FPGM0.3_2 0.9 0.3
pruning_scratch_resnet20 0 log/res20_norm0.6_FPGM0_3 0.6 0
}
run20_distill(){
pruning_scratch_resnet20_distill 0 log/res20_norm1_randn0.4_distill4_T10_alpha0.8 1 0.4 0.8
#pruning_scratch_resnet20_distill 0 log/res20_norm0.6_FPGM0_distill3_T10_alpha0.8 0.6 0 0.8
}


#run20_distill
run20_2
#run32
