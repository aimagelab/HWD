import argparse
import torch
import json
import copy
import warnings
from metrics import FReDScore, FIDScore, FontScore, BaseScore
from metrics import SilhouetteScore, CalinskiHarabaszScore, DaviesBouldinScore
from datasets import BaseDataset, CVLDataset, IAMDataset, LeopardiDataset, NorhandDataset
from datasets.transforms import fid_our_tranforms, fred_tranforms
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['cvl', 'iam', 'leopardi', 'norhand'])
    parser.add_argument('--score', type=str, required=True,
                        choices=['fid', 'fred', 'fred_mean', 'sfred', 'sfred_mean', 'font'])
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
    parser.add_argument('--skip_compute', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    if not args.skip_compute:
        if args.dataset == 'cvl':
            dataset = CVLDataset(args.path)
        elif args.dataset == 'iam':
            dataset = IAMDataset(args.path)
        elif args.dataset == 'leopardi':
            dataset = LeopardiDataset(args.path)
        elif args.dataset == 'norhand':
            dataset = NorhandDataset(args.path)
        else:
            raise ValueError('No dataset specified')
        assert isinstance(dataset, BaseDataset)

        if args.score == 'fid':
            score = FIDScore()
            dataset.transform = fid_our_tranforms
        elif args.score == 'fred':
            score = FReDScore(reduction=None)
            dataset.transform = fred_tranforms
        elif args.score == 'fred_mean':
            score = FReDScore()
            dataset.transform = fred_tranforms
        elif args.score == 'sfred':
            score = FReDScore(reduction=None, layers=1)
            dataset.transform = fred_tranforms
        elif args.score == 'sfred_mean':
            score = FReDScore(layers=1)
            dataset.transform = fred_tranforms
        elif args.score == 'font':
            score = FontScore()
            dataset.transform = fred_tranforms
        else:
            raise ValueError('No score specified')
        assert isinstance(score, BaseScore)

        if args.sort:
            dataset.sort(verbose=args.verbose)
        if not args.sort and args.batch_size > 1 and args.score in ['font', 'fred']:
            warnings.warn('Dataset is not sorted, the official score will be different')
        dataset.is_sorted = True

        good_samples = []
        for author_id in dataset.all_author_ids:
            dataset.author_ids = [author_id]
            tmp_db1, tmp_db2 = dataset.split(0.5)
            res = secure_compute(tmp_db1, tmp_db2, score, args.batch_size, args.verbose)

            if res is not None:
                good_samples.append(res)
            print(f'Done with {args.score} - {author_id}: {res}')
        print('Done with good samples')

        bad_samples = []
        intermediate_data = []
        for i, author_id in enumerate(dataset.all_author_ids):
            dataset.author_ids = [author_id, ]
            if len(dataset) == 0:
                print(f'No samples for {author_id}')
                continue
            dataset.transform = fid_our_tranforms
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

        with open(f'{args.dataset}_{args.score}', 'w') as f:
            json.dump(data, f)
        print('Done with saving')

    with open(f'{args.dataset}_{args.score}', 'r') as f:
        data = json.load(f)

    good_samples = data['good_samples']
    bad_samples = data['bad_samples']

    print(SilhouetteScore().distance(good_samples, bad_samples))
    print(CalinskiHarabaszScore().distance(good_samples, bad_samples))
    print(DaviesBouldinScore().distance(good_samples, bad_samples))

    kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)

    plt.hist(good_samples, **kwargs, color='g', label='Good')
    plt.hist(bad_samples, **kwargs, color='r', label='Bad')
    plt.gca().set(title=f'Frequency Histogram DATASET:{args.dataset} SCORE:{args.score}', ylabel='Frequency')
    plt.legend()

    plt.savefig(f'{args.dataset}_{args.score}.png')
    print('Done graphing')
    exit()
