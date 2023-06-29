compute=force
act=lazy

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fid --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fid_euc --verbose --compute $compute --act $act --device cuda --author_min 2

echo "---------------------------------------------"

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fved_imagenet_beginning --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score vont_imagenet_beginning --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fved_imagenet --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score vont_imagenet --verbose --compute $compute --act $act --device cuda --author_min 2

echo "---------------------------------------------"

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fved_beginning --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score vont_beginning --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score fved --verbose --compute $compute --act $act --device cuda --author_min 2

python make_graph.py --path /home/shared/datasets/IAM --dataset iam \
    --score vont --verbose --compute $compute --act $act --device cuda --author_min 2

