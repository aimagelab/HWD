import argparse
import signal
import csv
from make_graph import get_dataset, get_score


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def write_csv(path, data):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--score', type=str, required=True)
    parser.add_argument('--timer', type=int, default=20)
    args = parser.parse_args()

    try:
        dataset = get_dataset(args.dataset, args.path)
        score = get_score(args.score, dataset)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timer)
        print(args.score, args.dataset, score(dataset, dataset))
        signal.alarm(0)
    except TimeoutException:
        print('Timeout')
    except Exception as e:
        print('Error', e)
        write_csv('results.csv', [args.score, args.dataset, str(e)])
        exit()
    write_csv('results.csv', [args.score, args.dataset, 'OK'])
    print('Done')
