compute=lazy
act=lazy

#python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/cvl-database-1-1 --dataset cvl \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/cvl-database-1-1 --dataset cvl \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/KHATT_Arabic --dataset khatt \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/KHATT_Arabic --dataset khatt \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/Rimes --dataset rimes \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/Rimes --dataset rimes \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/Norhand --dataset norhand \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/Norhand --dataset norhand \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/CHS --dataset chs \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/CHS --dataset chs \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/BanglaWriting --dataset bangla \
#    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2
#
#python make_graph.py --path /home/shared/datasets/BanglaWriting --dataset bangla \
#    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2

python make_confidence_graph.py --path \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  leopardi \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  lam \
  --score fved --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/LAM \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/leopardi \
  --dataset \
  lam \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  leopardi \
  --score fved --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/LAM \
  --dataset \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  leopardi \
  lam \
  --score fved --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  saint_gall \
  leopardi \
  washington \
  icfhr14 \
  rodrigo \
  lam \
  --score fved --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  icfhr14 \
  leopardi \
  washington \
  saint_gall \
  rodrigo \
  lam \
  --score fved --verbose --compute $compute --act $act


python make_confidence_graph.py --path \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/LAM \
  --dataset \
  rodrigo \
  icfhr14 \
  leopardi \
  washington \
  saint_gall \
  lam \
  --score fved --verbose --compute $compute --act $act


python make_confidence_graph.py --path \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  leopardi \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  lam \
  --score fid_euc --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/LAM \
  --dataset \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  leopardi \
  lam \
  --score fid_euc --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  saint_gall \
  leopardi \
  washington \
  icfhr14 \
  rodrigo \
  lam \
  --score fid_euc --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/LAM \
  --dataset \
  icfhr14 \
  leopardi \
  washington \
  saint_gall \
  rodrigo \
  lam \
  --score fid_euc --verbose --compute $compute --act $act


python make_confidence_graph.py --path \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/leopardi \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/LAM \
  --dataset \
  rodrigo \
  icfhr14 \
  leopardi \
  washington \
  saint_gall \
  lam \
  --score fid_euc --verbose --compute $compute --act $act

python make_confidence_graph.py --path \
  /home/shared/datasets/LAM \
  /home/shared/datasets/Washington \
  /home/shared/datasets/SaintGall \
  /home/shared/datasets/ICFHR14 \
  /home/shared/datasets/Rodrigo \
  /home/shared/datasets/leopardi \
  --dataset \
  lam \
  washington \
  saint_gall \
  icfhr14 \
  rodrigo \
  leopardi \
  --score fid_euc --verbose --compute $compute --act $act