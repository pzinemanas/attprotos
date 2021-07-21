from dcase_models.data.datasets import URBAN_SED

import os
import argparse


available_datasets = {
    'URBAN_SED':  URBAN_SED,
}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. URBAN_SED)',
        default='UrbanSound8k'
    )
    parser.add_argument(
        '-dp', '--dataset_path', type=str,
        help='path to load the dataset',
        default='../datasets'
    )

    args = parser.parse_args()

    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    # Get and init dataset class
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, args.dataset)
    dataset = dataset_class(dataset_path)

    # Download dataset
    dataset.download()

    print('Done!')


if __name__ == "__main__":
    main()
