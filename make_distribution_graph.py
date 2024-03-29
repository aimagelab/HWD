import argparse
import torch
import json
import csv
import random
import numpy as np
from pathlib import Path
from metrics import *
from datasets import *
from metrics.distances import *
from metrics.backbones import *
from torch.cuda import OutOfMemoryError
from datasets.transforms import *
import matplotlib.pyplot as plt


def secure_compute(db1, db2, metric, batch_size, verbose=False):
    try:
        return metric(db1, db2, batch_size=batch_size, verbose=verbose)
    except ValueError as e:
        print(f'Failed to compute {metric.__class__.__name__} {e}')
        return None


def secure_compute_distance(data1, data2, score, verbose=False, batch_size=None, device='cuda'):
    kwargs = {'verbose': verbose}
    if batch_size is not None:
        kwargs['batch_size'] = batch_size
    try:
        data1 = data1.to(device)
        data2 = data2.to(device)
        assert len(data1) > 0 or len(data2) > 0
        return score.distance(data1, data2, **kwargs)
    except (ValueError, AssertionError) as e:
        print(f'Failed to compute {score.__class__.__name__} {len(data1)=} {len(data2)=} {e}')
        return None
    except OutOfMemoryError as e:
        print(f'Ouf of memory {score.__class__.__name__} retrying using cpu')
        return secure_compute_distance(data1, data2, score, verbose=verbose, batch_size=batch_size, device='cpu')


def get_dataset(dataset_name, path, **kwargs):
    if dataset_name == 'cvl':
        dataset = CVLDataset(path, **kwargs)
    elif dataset_name == 'iam':
        dataset = IAMDataset(path, **kwargs)
    elif dataset_name == 'iam_lines':
        dataset = IAMDataset(path, dataset_type='lines', **kwargs)
    elif dataset_name == 'leopardi':
        dataset = LeopardiDataset(path, **kwargs)
    elif dataset_name == 'norhand':
        dataset = NorhandDataset(path, **kwargs)
    elif dataset_name == 'rimes':
        dataset = RimesDataset(path, **kwargs)
    elif dataset_name == 'lam':
        dataset = LAMDataset(path, **kwargs)
    elif dataset_name == 'chs':
        dataset = CHSDataset(path, **kwargs)
    elif dataset_name == 'khatt':
        dataset = KHATTDataset(path, **kwargs)
    elif dataset_name == 'leo':
        dataset = LeopardiDataset(path, **kwargs)
    elif dataset_name == 'saint_gall':
        dataset = SaintGallDataset(path, **kwargs)
    elif dataset_name == 'washington':
        dataset = WashingtonDataset(path, **kwargs)
    elif dataset_name == 'rodrigo':
        dataset = RodrigoDataset(path, **kwargs)
    elif dataset_name == 'icfhr14':
        dataset = ICFHR14Dataset(path, **kwargs)
    elif dataset_name == 'chs':
        dataset = CHSDataset(path, **kwargs)
    elif dataset_name == 'bangla':
        dataset = BanglaWritingDataset(path, **kwargs)
    elif dataset_name.startswith('folder'):
        dataset = FolderDataset(path, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    assert isinstance(dataset, BaseDataset)
    return dataset


def get_score(score_name, dataset, device='cuda'): 
    kwargs = {'device': device}
    dataset.transform = fid_our_transforms
    if score_name == 'fid':
        score = FIDScore(**kwargs)
    elif score_name == 'hwd':
        score = HWDScore(**kwargs)
    elif score_name == 'hwd_hamming':
        score = HWDScore(**kwargs)
        score.distance = HammingDistance(threshold=0.005)
    elif score_name == 'hwd_mahalanobis':
        score = HWDScore(**kwargs)
        score.distance = MahalanobisDistance()
    elif score_name == 'hwd_cosine':
        score = HWDScore(**kwargs)
        score.distance = CosineDistance()
    elif score_name == 'fid_rand':
        score = FIDScore(**kwargs)
        score.backbone = InceptionV3Backbone(url=None)  
        score.backbone.to(device)
    elif score_name == 'iv3_frechet':
        backbone = InceptionV3Backbone(url='https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400.pth')
        distance = FrechetDistance()
        transforms = Compose([
                ResizeHeight(32),
                ToTensor(),
                ToInceptionV3Input(size=299),
            ])
        score = BaseScore(backbone, distance, transforms, **kwargs)
    elif score_name == 'iv3_euclidean':
        backbone = InceptionV3Backbone(url='https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400.pth')
        distance = EuclideanDistance()
        transforms = Compose([
                ResizeHeight(32),
                ToTensor(),
                ToInceptionV3Input(size=299),
            ])
        score = BaseScore(backbone, distance, transforms, **kwargs)
    elif score_name == 'custom_vgg_rand_begin_euc':
        backbone = InceptionV3Backbone(url='https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400.pth')
        distance = EuclideanDistance()
        transforms = Compose([
                ResizeHeight(32),
                ToTensor(),
                ToInceptionV3Input(size=299),
            ])
        score = BaseScore(backbone, distance, transforms, **kwargs)
    elif score_name == 'iv3_cosine':
        backbone = InceptionV3Backbone(url='https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400.pth')
        distance = CosineDistance()
        transforms = Compose([
                ResizeHeight(32),
                ToTensor(),
                ToInceptionV3Input(size=299),
            ])
        score = BaseScore(backbone, distance, transforms, **kwargs)
    elif score_name.startswith('custom_'):
        _, backbone_name, pretraining, transforms_name, distance_name = score_name.split('_')

        if backbone_name == 'vgg':
            rand_url = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400_random.pth'
            font2_url = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth'
            backbone = VGG16Backbone(url=rand_url) if pretraining == 'rand' else VGG16Backbone(url=font2_url)
        elif backbone_name == 'iv3':
            rand_url = 'https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400_random.pth'
            backbone = InceptionV3Backbone(url=rand_url) if pretraining == 'rand' else InceptionV3Backbone()
        
        if distance_name == 'euc':
            distance = EuclideanDistance()
        elif distance_name == 'fre':
            distance = FrechetDistance()
        elif distance_name == 'cos':
            distance = CosineDistance()
        
        if transforms_name == 'begin':
            transforms = Compose([
                ResizeHeight(32),
                CropStartSquare(),
                ToTensor(),
            ])
        elif transforms_name == 'whole':
            transforms = Compose([
                ResizeHeight(32),
                ToTensor(),
            ])

        score = BaseScore(backbone, distance, transforms, **kwargs)
    else:
        raise ValueError(f'Unknown score {score_name}')
    assert isinstance(score, BaseScore)
    return score


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--csv_path', type=str, default='results.csv', help='Path to the csv file')
    parser.add_argument('--results_path', type=str, default='results', help='Path to the results folder')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--score', type=str, required=True)
    parser.add_argument('--compute', type=str, default='lazy', choices=['force', 'lazy', 'none'])
    parser.add_argument('--act', type=str, default='lazy', choices=['force', 'lazy'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--author_min', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--strict_positive', action='store_true', default=False)
    args = parser.parse_args()

    args.results_path = Path(args.results_path)
    args.results_path.mkdir(exist_ok=True)

    pickle_path = args.results_path / f'processed_{args.dataset}_{args.score}.pkl'
    json_path = args.results_path / f'{args.dataset}_{args.score}.json'
    csv_path = args.results_path / args.csv_path

    if args.compute == 'force' or (args.compute == 'lazy' and not json_path.exists()):
        set_seed(args.seed)
        dataset = get_dataset(args.dataset, args.path)
        score = get_score(args.score, dataset, args.device)

        author_ids = dataset.author_ids
        author_sizes = [sum(label == author for label in dataset.labels) for author in author_ids]
        author_ids = sorted({author for author, size in zip(author_ids, author_sizes) if size > args.author_min})

        kwargs = {'dataset': dataset, 'verbose': True}
        if args.batch_size is not None:
            kwargs['batch_size'] = args.batch_size

        if args.act == 'force' or (args.act == 'lazy' and not pickle_path.exists()):
            processed_dataset = score.digest(**kwargs)
            processed_dataset.save(pickle_path)
        else:
            print(f'Loading activations {pickle_path}...')
            processed_dataset = ProcessedDataset.load(pickle_path)

        good_samples = []
        for i, author_id in enumerate(author_ids, 1):
            tmp_db1, tmp_db2 = processed_dataset[author_id].split(0.5)
            res = secure_compute_distance(tmp_db1, tmp_db2, score, True, args.batch_size, args.device)
            if res is not None:
                good_samples.append(res)
            print(f'\rComputing GOOD {i}/{len(author_ids)} ', end='', flush=True)
        print('OK')

        bad_samples = []
        for i, author_id in enumerate((author_ids), 1):
            # other_authors = author_ids - {author_id}
            other_authors = author_ids[i % len(author_ids)]
            tmp_db1 = processed_dataset[author_id]
            tmp_db2 = processed_dataset[other_authors]
            res = secure_compute_distance(tmp_db1, tmp_db2, score, True, args.batch_size, args.device)
            if res is not None:
                bad_samples.append(res)
            print(f'\rComputing BAD {i}/{len(author_ids)} ', end='', flush=True)
        print('OK')

        data = {
            'good_samples': good_samples,
            'bad_samples': bad_samples,
            'dataset': args.dataset,
            'score': args.score,
        }

        with open(json_path, 'w') as f:
            json.dump(data, f)
        print('Done with saving')

    if not json_path.exists():
        print(f'File {json_path} does not exist')
        exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    good_samples = np.array(data['good_samples'])
    bad_samples = np.array(data['bad_samples'])

    good_samples = good_samples[~np.isnan(good_samples)]
    bad_samples = bad_samples[~np.isnan(bad_samples)]

    # discard negative samples
    if args.strict_positive:
        good_samples = good_samples[good_samples >= 0]
        bad_samples = bad_samples[bad_samples >= 0]

    silhouette_score = SilhouetteScore().distance(good_samples, bad_samples)
    calinski_harabasz_score = CalinskiHarabaszScore().distance(good_samples, bad_samples)
    davies_bouldin_score = DaviesBouldinScore().distance(good_samples, bad_samples)
    gray_zone_score = GrayZoneScore().distance(good_samples, bad_samples)
    equal_error_rate_score = EqualErrorRateScore().distance(good_samples, bad_samples)
    vit_score = VITScore().distance(good_samples, bad_samples)

    if not csv_path.exists():
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'score', 'batch_size', 'seed', 'silhouette', 'calinski', 'davies',
                             'gray_zone', 'miss_rate', 'vit'])

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [args.dataset, args.score, args.batch_size, args.seed, silhouette_score, calinski_harabasz_score,
             davies_bouldin_score, gray_zone_score, equal_error_rate_score, vit_score])

    kwargs = dict(alpha=0.5, density=True, stacked=True)
    bins = np.histogram(np.hstack((good_samples, bad_samples)), bins=40)[1]  # get the bin edges

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'DATASET:{args.dataset} SCORE:{args.score}')

    axs[0, 0].hist(good_samples, bins, **kwargs, color='g', label='Good')
    axs[0, 0].hist(bad_samples, bins, **kwargs, color='r', label='Bad')
    axs[0, 0].legend()

    axs[1, 0].hist(good_samples, bins, **kwargs, color='g', label='Good')
    axs[1, 0].hist(bad_samples, bins, **kwargs, color='r', label='Bad')
    axs[1, 0].set_xscale('log')

    columns = ('Metric', 'Value')
    cell_text = [
        ['Silhouette', f'{silhouette_score:.3f}'],
        ['Calinski-Harabasz', f'{calinski_harabasz_score:.3f}'],
        ['Davies-Bouldin', f'{davies_bouldin_score:.3f}'],
        ['Gray Zone', f'{gray_zone_score:.3f}'],
        ['Equal Error Rate', f'{equal_error_rate_score:.3f}'],
        ['VIT', f'{vit_score:.3f}'],
    ]

    axs[0, 1].axis('tight')
    axs[0, 1].axis('off')
    axs[0, 1].table(cellText=cell_text, colLabels=columns, loc='center')

    kwargs = dict(alpha=0.1, marker="|")
    axs[1, 1].scatter(good_samples, [0] * len(good_samples), color='g', label='Good', **kwargs)
    axs[1, 1].scatter(bad_samples, [0] * len(bad_samples), color='r', label='Bad', **kwargs)
    # axs[1, 1].set_xscale('log')
    axs[1, 1].set_yticks([], [])
    # set plt in
    axs[1, 1].legend()

    plt.savefig(args.results_path / f'{args.dataset}_{args.score}.png', dpi=900)
    plt.close()

    kwargs = dict(alpha=0.5, density=True, stacked=True)
    plt.hist(good_samples, bins, **kwargs, color='g', label='Good')
    plt.hist(bad_samples, bins, **kwargs, color='r', label='Bad')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(args.results_path / f'{args.dataset}_{args.score}.pdf')

    print('Done graphing')
    print(f'Overlap: {gray_zone_score:.3f}')
    print(f'Equal Error Rate: {equal_error_rate_score:.3f}')
    exit()
