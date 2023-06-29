import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from make_graph import get_dataset, get_score, set_seed, secure_compute_distance
from metrics import ProcessedDataset


def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', type=str, required=True)
    parser.add_argument('--path', nargs='+', type=str, required=True)
    parser.add_argument('--score', type=str, required=True)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--results_path', type=str, default='results', help='Path to the results folder')
    parser.add_argument('--compute', type=str, default='lazy', choices=['lazy', 'force'])
    parser.add_argument('--act', type=str, default='lazy', choices=['lazy', 'force'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_samples', type=int, default=10)
    args = parser.parse_args()

    assert len(args.dataset) == len(args.path), 'Number of datasets and paths must be the same'
    assert len(args.dataset) > 1, 'At least two datasets must be provided'
    datasets = [get_dataset(dataset, path) for dataset, path in zip(args.dataset, args.path)]
    dataset_main = datasets[0]
    score = get_score(args.score, dataset_main)
    for dataset in datasets:
        get_score(args.score, dataset)

    args.results_path = Path(args.results_path)
    args.results_path.mkdir(exist_ok=True)

    pickle_paths = [args.results_path / f'processed_{d}_{args.score}.pkl' for d in args.dataset]
    json_paths = [args.results_path / f'conf_multiple_{args.score}_{args.dataset[0]}_{d}.json' for d in args.dataset]

    if args.compute == 'force' or (args.compute == 'lazy' and not all(j.exists() for j in json_paths)):
        set_seed(args.seed)

        milestones = [len(dataset_main), ]
        while milestones[-1] > args.min_samples:
            milestones.append(milestones[-1] // 2)
        print(f'Milestones: {milestones}')

        results = {d: {m: [] for m in milestones} for d, j in zip(args.dataset, json_paths) if not j.exists()}

        kwargs = {'verbose': args.verbose}
        if args.batch_size is not None:
            kwargs['batch_size'] = args.batch_size


        def get_processed_dataset(path, dataset):
            if args.act == 'force' or (args.act == 'lazy' and not path.exists()):
                tmp_dataset = score.digest(dataset=dataset, **kwargs)
                tmp_dataset.save(path)
            else:
                if args.verbose: print(f'Loading activations {path}...')
                tmp_dataset = ProcessedDataset.load(path)
            return tmp_dataset


        processed_dataset_main = get_processed_dataset(pickle_paths[0], datasets[0])
        processed_datasets = [get_processed_dataset(p, d) for p, d, j in zip(pickle_paths, datasets, json_paths) if not j.exists()]
        processed_datasets_names = [dataset_name for dataset_name, j in zip(args.dataset, json_paths) if not j.exists()]
        processed_datasets = [pd.to(args.device) for pd in processed_datasets]
        processed_dataset_main = processed_dataset_main.to(args.device)

        for milestone in milestones:
            print(f'Processing milestone: {milestone}')
            for i in range(args.repetitions):
                sub_dataset = processed_dataset_main.subset(milestone)


                def compute(pd, dataset_name):
                    dist = secure_compute_distance(pd, sub_dataset, score, args.verbose, args.batch_size, args.device)
                    if dist is not None:
                        results[dataset_name][milestone].append(dist)
                    return dist


                for pd, pd_name in zip(processed_datasets, processed_datasets_names):
                    dist = compute(pd, pd_name)
                    print(f'\r{i + 1}/{args.repetitions} {pd_name: <10} {dist} ', end='', flush=True)
                if milestone == milestones[0]:
                    break
            print('OK')

        for json_path, dataset_name in ((j, n) for j, n in zip(json_paths, args.dataset) if not j.exists()):
            with open(json_path, 'w') as f:
                json.dump(results[dataset_name], f)
        print('Saved results')

    results = {}
    for json_path, dataset_name in zip(json_paths, args.dataset):
        with open(json_path, 'r') as f:
            results[dataset_name] = json.load(f)

    x = np.array(sorted(int(k) for k in results[args.dataset[0]].keys()))
    dst_path = f'{args.results_path}/conf_{args.score}_{"_".join(args.dataset)}.pdf'
    db_colors = {
        'lam': '#3498db',
        'leopardi': '#ff993e',
        'washington': '#58d68d',
        'saint_gall': '#c0392b',
        'icfhr14': '#9b59b6',
        'rodrigo': '#f1c40f',
    }

    y = [np.array([np.mean(results[d][str(k)]) for k in x]) for d in args.dataset]
    y_low = [np.array([np.percentile(results[d][str(k)], 25) for k in x]) for d in args.dataset]
    y_high = [np.array([np.percentile(results[d][str(k)], 75) for k in x]) for d in args.dataset]

    fig, axs = plt.subplots(1)
    # fig.suptitle(f'{args.score} - {args.dataset}')

    for d, y_, y_low_, y_high_ in zip(args.dataset, y, y_low, y_high):
        axs.plot(x, y_, label=d, color=db_colors[d])
        axs.fill_between(x, y_low_, y_high_, alpha=.1, color=db_colors[d])
    axs.set_xscale('log')
    plt.tight_layout()
    # axs.legend()

    # axs[1, 0].plot(x, y_good, label='good')
    # axs[1, 0].fill_between(x, y_low_good, y_high_good, color='b', alpha=.1)
    # axs[1, 0].plot(x, y_bad, label='bad')
    # axs[1, 0].fill_between(x, y_low_bad, y_high_bad, color='r', alpha=.1)
    # # axs[1, 0].set_yscale('log')
    # axs[1, 0].set_xscale('log')
    #
    # axs[0, 1].plot(x, y_bad, label='bad')
    # axs[0, 1].fill_between(x, y_low_bad, y_high_bad, color='r', alpha=.1)
    # # axs[0, 0].set_yscale('log')
    # axs[0, 1].set_xscale('log')

    # axs[1, 0].plot(x, y / y.max())
    # axs[1, 0].set_xscale('log')
    #
    # var = np.array([np.var(results[str(k)]) for k in x])
    # axs[0, 1].plot(x, var)
    # axs[0, 1].set_yscale('log')
    #
    # axs[1, 1].plot(x, var)

    plt.savefig(dst_path)

    print('Done')
