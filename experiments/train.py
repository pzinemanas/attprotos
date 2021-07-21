import os
import argparse
import sys
import numpy as np

sys.path.append('../')
# Datasets
from dcase_models.data.datasets import URBAN_SED

# Models
from dcase_models.model.models import SB_CNN_SED, MLP
from attprotos.model import AttProtos
from attprotos.layers import PrototypeReconstruction

# Features
from dcase_models.data.features import MelSpectrogram
from attprotos.features import Openl3

from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json
from dcase_models.util.files import mkdir_if_not_exists, save_pickle
from dcase_models.util.data import evaluation_setup


available_models = {
    'AttProtos': AttProtos,
    'SB_CNN_SED': SB_CNN_SED,
    'MLP': MLP,
}

available_features = {
    'MelSpectrogram':  MelSpectrogram,
    'Openl3': Openl3
}

available_datasets = {
    'URBAN_SED': URBAN_SED,
}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='UrbanSound8k'
    )
    parser.add_argument(
        '-f', '--features', type=str,
        help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
        default='MelSpectrogram'
    )

    parser.add_argument(
        '-m', '--model', type=str,
        help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)')

    parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                        default='fold1')

    parser.add_argument(
        '-mp', '--models_path', type=str,
        help='path to load the trained model',
        default='./'
    )
    parser.add_argument(
        '-dp', '--dataset_path', type=str,
        help='path to load the dataset',
        default='./'
    )

    parser.add_argument('--c', dest='continue_training', action='store_true')
    parser.set_defaults(continue_training=False)

    parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                        default='0')

    args = parser.parse_args()

    # only use one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible

    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    if args.features not in available_features:
        raise AttributeError('Features not available')

    model_name = args.model
    if model_name not in available_models:
        model_name = args.model.split('/')[0]
        if model_name not in available_models:
            raise AttributeError('Model not available')

    # Model paths
    model_folder = os.path.join(args.models_path, args.dataset, args.model)

    # Get parameters
    parameters_file = os.path.join(model_folder, 'config.json')
    params = load_json(parameters_file)

    params_features = params['features'][args.features]
    if 'pad_mode' in params_features:
        if params_features['pad_mode'] == 'none':
            params_features['pad_mode'] = None
    params_dataset = params['datasets'][args.dataset]
    params_model = params['models'][model_name]

    # Get and init dataset class
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
    print(dataset_path)
    dataset = dataset_class(dataset_path)

    if args.fold_name not in dataset.fold_list:
        raise AttributeError('Fold not available')

    # Get and init feature class
    features_class = available_features[args.features]
    features = features_class(**params_features)

    print('Features shape: ', features.get_shape(10.0))

    if not features.check_if_extracted(dataset):
        print('Extracting features ...')
        features.extract(dataset)
        print('Done!')


    folds_train, folds_val, folds_test = evaluation_setup(
        args.fold_name, dataset.fold_list,
        params_dataset['evaluation_mode'],
        use_validate_set=True
    )

    if model_name == 'AttProtos':
        outputs = ['annotations', features, 'zeros', 'zeros']
    else:
        outputs = 'annotations'

    data_gen_train = DataGenerator(
        dataset, features, folds=folds_train,
        batch_size=params['train']['batch_size'],
        shuffle=True, train=True, scaler=None,
        outputs=outputs
    )

    scaler = Scaler(normalizer=params_model['normalizer'])

    print('Fitting scaler ...')
    if args.model == 'AttProtos':
        scaler_outputs = Scaler(
            normalizer=[None, params_model['normalizer'], None, None])
        scaler_outputs.fit(data_gen_train, inputs=False)
        data_gen_train.set_scaler_outputs(scaler_outputs)
    else:
        scaler_outputs = None

    scaler.fit(data_gen_train)
    data_gen_train.set_scaler(scaler)
    print('Done!')

    # Pass scaler to data_gen_train to be used when data
    # loading

    data_gen_val = DataGenerator(
        dataset, features, folds=folds_val,
        batch_size=params['train']['batch_size'],
        shuffle=False, train=False, scaler=scaler
    )

    # Define model
    features_shape = features.get_shape()
    if len(features_shape) > 2:
        n_frames_cnn = features_shape[1]
        n_freq_cnn = features_shape[2]
    else:
        n_freq_cnn = features_shape[1]
    n_classes = len(dataset.label_list)

    model_class = available_models[model_name]

    if args.dataset in ['URBAN_SED', 'TUTSoundEvents2017', 'MAVD']:
        metrics = ['sed']
    else:
        metrics = ['classification']

    # Set paths
    exp_folder = os.path.join(model_folder, args.fold_name)
    mkdir_if_not_exists(exp_folder, parents=True)

    if args.continue_training:
        model_container = model_class(
            model=None, model_path=exp_folder,
            custom_objects={
                'PrototypeReconstruction': PrototypeReconstruction,
            }
        )
        model_container.load_model_weights(exp_folder)
        params_model['train_arguments']['init_last_layer'] = 0
    else:
        if args.model == 'MLP':
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames=None, n_freqs=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        else:
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        model_container.save_model_json(exp_folder)

    model_container.model.summary()

    # Save model json and scaler
    save_pickle(scaler, os.path.join(exp_folder, 'scaler.pickle'))

    data_gen_train = data_gen_train.get_data()

    if args.model not in ['MLP', 'SB_CNN_SED']:
        params_model['train_arguments']['loss_classification'] = 'binary_crossentropy'
    else:
        params_model['train_arguments']['losses'] = 'binary_crossentropy'

    # Train model
    model_container.train(
        data_gen_train, data_gen_val,
        label_list=dataset.label_list,
        weights_path=exp_folder, **params['train'],
        sequence_time_sec=params_features['sequence_hop_time'],
        **params_model['train_arguments']
    )


if __name__ == "__main__":
    main()
