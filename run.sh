#!/bin/sh
task_list=(task8 task9 task10 task11)
log_list=(./log/bartpho-sum-vinewsqa ./log/vit5-sum-vinewsqa ./log/bartpho-sum-viquad ./log/vit5-sum-viquad)

for (( i=0; i<${#task_list[*]}; ++i)); do
    echo "training ${task_list[$i]}"
    nohup python3 pytorch_trainer.py --gpu 1 --task ${task_list[$i]} > ${log_list[$i]}/${task_list[$i]}.txt
done
