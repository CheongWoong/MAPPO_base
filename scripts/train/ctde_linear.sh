#!/bin/sh

env_id=$1
filename=$(basename $0)
filename="${filename%.*}"
total_timesteps=10000000
seed=0

python \
    -m src.train \
    --critic-type centralized \
    --critic-num-heads 0 \
    --actor-type decentralized \
    --actor-num-heads 0 \
    --env-id $env_id \
    --total-timesteps $total_timesteps \
    --seed $seed \
    --exp-name $env_id"/"$filename"_"$total_timesteps"_"$seed