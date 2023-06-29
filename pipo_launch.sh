#!/usr/bin/env bash

metrics=(
    'fid'
    'kid'
#    'font'
#    'fred_4'
#    'fred_3'
#    'fred_2'
#    'fred_1'
#    'fred_mean_4'
#    'fred_mean_3'
#    'fred_mean_2'
#    'fred_mean_1'
#    'kred_4'
#    'kred_3'
#    'kred_2'
#    'kred_1'
#    'kred_mean_4'
#    'kred_mean_3'
#    'kred_mean_2'
#    'kred_mean_1'
#    'fved_4'
#    'fved_3'
##    'fved_2'
##    'fved_1'
##    'fved_0'
#    'fved_mean_4'
#    'fved_mean_3'
#    'fved_mean_2'
#    'fved_mean_1'
#    'fved_mean_0'
#    'kved_4'
#    'kved_3'
##    'kved_2'
##    'kved_1'
#    'kved_mean_4'
#    'kved_mean_3'
#    'kved_mean_2'
#    'kved_mean_1'
#    'tred_mean'
#    'tved_mean'
    'vont'
#    'vont_3'
##    'vont_2'
##    'vont_1'
#    'vont_mean_4'
#    'vont_mean_3'
##    'vont_mean_2'
##    'vont_mean_1'
    )

cd /home/vpippi/ICIAP_FID

for i in ${!metrics[@]}; do
    echo ${metrics[$i]}
    python make_graph.py --path /home/shared/datasets/IAM --dataset iam --score ${metrics[$i]} --verbose
    python make_graph.py --path /home/shared/datasets/cvl-database-1-1 --dataset cvl --score ${metrics[$i]} --verbose
    python make_graph.py --path /home/shared/datasets/KHATT_Arabic --dataset khatt --score ${metrics[$i]} --verbose
    python make_graph.py --path /home/shared/datasets/Norhand --dataset norhand --score ${metrics[$i]} --verbose
    python make_graph.py --path /home/shared/datasets/Rimes --dataset rimes --score ${metrics[$i]} --verbose
    python make_graph.py --path /home/shared/datasets/BanglaWriting --dataset bangla --score ${metrics[$i]} --verbose
done