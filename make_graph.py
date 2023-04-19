import argparse
import torch
import json
import copy
import csv
import random
import warnings
import datetime
from pathlib import Path
from metrics import *
from datasets import *
from datasets.transforms import fid_our_transforms, fred_transforms
import matplotlib.pyplot as plt


def secure_compute(db1, db2, metric, batch_size, verbose=False):
    try:
        return metric(db1, db2, batch_size=batch_size, verbose=verbose)
    except ValueError as e:
        print(f'Failed to compute {metric.__class__.__name__} {e}')
        return None


def secure_compute_distance(data1, data2, metric, verbose=False):
    try:
        return metric.distance(data1, data2, verbose=verbose)
    except ValueError as e:
        print(f'Failed to compute {metric.__class__.__name__} {e}')
        return None


def get_dataset(dataset_name, path):
    if dataset_name == 'cvl':
        dataset = CVLDataset(path)
    elif dataset_name == 'iam':
        dataset = IAMDataset(path)
    elif dataset_name == 'leopardi':
        dataset = LeopardiDataset(path)
    elif dataset_name == 'norhand':
        dataset = NorhandDataset(path)
    elif dataset_name == 'rimes':
        dataset = RimesDataset(path)
    elif dataset_name == 'lam':
        dataset = LAMDataset(path)
    elif dataset_name == 'chs':
        dataset = CHSDataset(path)
    elif dataset_name == 'hkr':
        dataset = HKRDataset(path)
    elif dataset_name == 'khatt':
        dataset = KHATTDataset(path)
    elif dataset_name == 'leo':
        dataset = LeopardiDataset(path)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    assert isinstance(dataset, BaseDataset)
    return dataset

def get_score(score_name, dataset):
    if score_name == 'fid':
        score = FIDScore()
        dataset.transform = fid_our_transforms
    elif score_name.startswith('fred_mean'):
        layers = score_name.split('_')[-1]
        layers = int(layers) if layers.isdigit() else 4
        score = FReDScore(layers=layers)
        dataset.transform = fred_transforms
    elif score_name.startswith('fred'):
        layers = score_name.split('_')[-1]
        layers = int(layers) if layers.isdigit() else 4
        score = FReDScore(layers=layers, reduction=None)
        dataset.transform = fred_transforms
    elif score_name.startswith('kred_mean'):
        layers = score_name.split('_')[-1]
        layers = int(layers) if layers.isdigit() else 4
        score = KReDScore(layers=layers)
        dataset.transform = fred_transforms
    elif score_name.startswith('kred'):
        layers = score_name.split('_')[-1]
        layers = int(layers) if layers.isdigit() else 4
        score = KReDScore(layers=layers, reduction=None)
        dataset.transform = fred_transforms
    elif score_name == 'font':
        score = FontScore()
        dataset.transform = fred_transforms
    elif score_name == 'kid':
        score = KIDScore()
        dataset.transform = fred_transforms
    else:
        raise ValueError(f'Unknown score {args.score}')
    assert isinstance(score, BaseScore)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--csv_path', type=str, default='results.csv', help='Path to the csv file')
    parser.add_argument('--results_path', type=str, default='results', help='Path to the results folder')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--score', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
    parser.add_argument('--skip_compute', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.results_path = Path(args.results_path)
    args.results_path.mkdir(exist_ok=True)

    if not args.skip_compute:
        dataset = get_dataset(args.dataset, args.path)
        score = get_score(args.score, dataset)

        if args.sort:
            dataset.sort(verbose=args.verbose)
        if not args.sort and args.batch_size > 1 and args.score in ['font', 'fred']:
            warnings.warn('Dataset is not sorted, the official score will be different')
        dataset.is_sorted = True

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        good_samples = []
        for _ in range(len(dataset.all_author_ids) // 2):
            for author_id in dataset.all_author_ids:
                dataset.author_ids = [author_id]
                tmp_db1, tmp_db2 = dataset.split(0.5)
                print(f'\rComputing {args.score} - {author_id} {len(dataset): >4d}: ', end='')
                if len(tmp_db1) == 0 or len(tmp_db2) == 0:
                    print('No enough samples')
                    continue
                res = secure_compute(tmp_db1, tmp_db2, score, args.batch_size, args.verbose)

                if res is not None:
                    good_samples.append(res)
                print(f'{res:.03f}')
        print('Done with good samples')

        bad_samples = []
        intermediate_data = []
        for i, author_id in enumerate(dataset.all_author_ids):
            dataset.author_ids = [author_id, ]
            if len(dataset) == 0:
                print(f'No samples for {author_id}')
                continue
            data = score.digest(dataset, batch_size=args.batch_size, verbose=args.verbose)
            intermediate_data.append(data)
            print(f'\rComputing activations {i + 1}/{len(dataset.all_author_ids)} ', end='', flush=True)
        print('OK')

        if hasattr(score, 'distance_batch'):
            bad_samples = score.distance_batch(intermediate_data, verbose=args.verbose)
        else:
            total_combinations = len(dataset.all_author_ids) * (len(dataset.all_author_ids) - 1) // 2
            count = 0
            for idx in range(len(dataset.all_author_ids)):
                for idy in range(idx + 1, len(dataset.all_author_ids)):
                    res = secure_compute_distance(intermediate_data[idx], intermediate_data[idy], score, args.verbose)
                    if res is not None:
                        bad_samples.append(res)

                    count += 1
                    print(f'\rComputing score {count}/{total_combinations} ', end='', flush=True)
            print('OK')
        print('Done with bad samples')

        data = {
            'good_samples': good_samples,
            'bad_samples': bad_samples,
            'dataset': args.dataset,
            'score': args.score,
        }

        with open(args.results_path / f'{args.dataset}_{args.score}.json', 'w') as f:
            json.dump(data, f)
        print('Done with saving')

    with open(args.results_path / f'{args.dataset}_{args.score}.json', 'r') as f:
        data = json.load(f)

    good_samples = data['good_samples']
    bad_samples = data['bad_samples']

    silhouette_score = SilhouetteScore().distance(good_samples, bad_samples)
    calinski_harabasz_score = CalinskiHarabaszScore().distance(good_samples, bad_samples)
    davies_bouldin_score = DaviesBouldinScore().distance(good_samples, bad_samples)

    with open(args.results_path / args.csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ args.dataset, args.score, args.batch_size, args.seed, args.sort,
                         silhouette_score, calinski_harabasz_score, davies_bouldin_score])

    kwargs = dict(alpha=0.5, density=True, stacked=True)

    plt.hist(good_samples, **kwargs, color='g', label='Good')
    plt.hist(bad_samples, **kwargs, color='r', label='Bad')
    plt.gca().set(title=f'DATASET:{args.dataset} SCORE:{args.score}\n'
                        f'SIL={silhouette_score:.3f} CAL={calinski_harabasz_score:.3f} '
                        f'DAV={davies_bouldin_score:.3f}', ylabel='Frequency')
    plt.legend()

    plt.savefig(args.results_path / f'{args.dataset}_{args.score}.png')
    print('Done graphing')
    exit()
