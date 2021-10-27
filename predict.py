# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md
	
import cog
import sys
sys.path.append('../')
from attprotos.model import AttProtos
from attprotos.layers import PrototypeReconstruction
from dcase_models.data.datasets import URBAN_SED
from dcase_models.data.features import MelSpectrogram
from dcase_models.util.files import load_json, load_pickle, save_pickle
import os
import tensorflow as tf
graph = tf.get_default_graph()
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import numpy as np


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        # Model paths
        models_path = './experiments'
        dataset = 'URBAN_SED'
        model_name = 'AttProtos'
        fold = 'test_1'
        features_name = 'MelSpectrogram'
        model_folder = os.path.join(models_path, dataset, model_name)

        # Get parameters
        parameters_file = os.path.join(model_folder, 'config.json')
        params = load_json(parameters_file)

        params_features = params['features'][features_name]
        if 'pad_mode' in params_features:
            if params_features['pad_mode'] == 'none':
                params_features['pad_mode'] = None
        params_dataset = params['datasets'][dataset]
        params_model = params['models'][model_name]

        exp_folder = os.path.join(model_folder, fold)

        scaler_file = os.path.join(exp_folder, 'scaler.pickle')
        self.scaler = load_pickle(scaler_file)

        # Get and init feature class
        self.features_extractor = MelSpectrogram(**params_features)

        kwargs = {
            'custom_objects': {
                'PrototypeReconstruction': PrototypeReconstruction,
            }
        }
        self.dataset = URBAN_SED('./')
        features_shape = self.features_extractor.get_shape()
        n_frames_cnn = features_shape[1]
        n_freq_cnn = features_shape[2]
        n_classes = len(self.dataset.label_list)
        self.model = AttProtos(
            model=None, model_path=None, n_classes=n_classes,
            n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
            metrics=['sed'],
            **params_model['model_arguments']
        )

        self.model.load_model_weights(exp_folder)


    @cog.input("audio_file", type=cog.Path, help="Audio file")
    def predict(self, audio_file):
        """Run a single prediction on the model"""
        features = self.features_extractor.calculate(audio_file)
        features = self.scaler.transform(features)
        with graph.as_default():
            pred = self.model.model.predict(features)[0]
            alpha = self.model.model_encoder_mask.predict(features)

        images = []
        for class_ix in np.arange(len(self.dataset.label_list)):

            time_hops = np.argwhere(pred[:, class_ix]*(pred[:, class_ix]>0.5)!=0)

            logits = pred
            mask_logits = np.zeros_like(logits)

            mask_logits[time_hops, class_ix] = 1

            logits = logits*mask_logits

            w = self.model.model.get_layer('dense').get_weights()[0]

            grad = logits.dot(w.T)
            grad = np.reshape(grad, (10, 32, 15))

            alpha2 = np.sum(alpha, axis=1)

            alpha_grad = grad*alpha2
            alpha_grad = alpha_grad*(alpha_grad>0)


            data = np.concatenate(features, axis=0)
            saliency = np.zeros_like(data)

            for th in time_hops:
                energy = np.sum(alpha_grad[th[0]]**2, axis=0)
                profile = alpha_grad[th[0], :, np.argmax(energy)]

                profile_extend = np.interp(np.arange(128), np.arange(32)*4, profile)

                profile_extend = np.convolve(profile_extend, [1/32]*32, mode='same')

                saliency[th[0]*20:(th[0] + 1)*20] = profile_extend

            masked_data = 2*((data+1)*saliency/2) - 1
            images.append(np.expand_dims(masked_data, 0))

        images = np.concatenate(images, axis=0)
        data = np.concatenate(features, axis=0)

        plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

        plt.figure(figsize=(10,20))
        plt.subplot(6,2,1)
        plt.imshow(data.T, origin='lower')
        plt.title('mel-spectrogram')
        plt.ylabel('mel filter index')
        plt.subplot(6,2,2)
        plt.imshow((pred*(pred>0.5)).T, origin='lower')
        plt.yticks(np.arange(len(self.dataset.label_list)), self.dataset.label_list)
        plt.title('activations')
        #plt.xlabel('hop time')
        plt.subplot(6,2,3)
        plt.imshow(images[0].T, origin='lower')
        plt.title(self.dataset.label_list[0])
        plt.ylabel('mel filter index')
        #plt.xlabel('hop time')
        plt.subplot(6,2,4)
        plt.imshow(images[1].T, origin='lower')
        plt.title(self.dataset.label_list[1])
        plt.subplot(6,2,5)
        plt.imshow(images[2].T, origin='lower')
        plt.title(self.dataset.label_list[2])
        plt.ylabel('mel filter index')
        plt.subplot(6,2,6)
        plt.imshow(images[3].T, origin='lower')
        plt.title(self.dataset.label_list[3])
        plt.subplot(6,2,7)
        plt.imshow(images[4].T, origin='lower')
        plt.title(self.dataset.label_list[4])
        plt.ylabel('mel filter index')
        plt.subplot(6,2,8)
        plt.imshow(images[5].T, origin='lower')
        plt.title(self.dataset.label_list[5])
        plt.subplot(6,2,9)
        plt.imshow(images[6].T, origin='lower')
        plt.title(self.dataset.label_list[6])
        plt.ylabel('mel filter index')
        plt.subplot(6,2,10)
        plt.imshow(images[7].T, origin='lower')
        plt.title(self.dataset.label_list[7])
        plt.subplot(6,2,11)
        plt.imshow(images[8].T, origin='lower')
        plt.title(self.dataset.label_list[8])
        plt.ylabel('mel filter index')
        plt.xlabel('hop time')
        plt.subplot(6,2,12)
        plt.imshow(images[9].T, origin='lower')
        plt.title(self.dataset.label_list[9])
        plt.ylabel('mel filter index')
        plt.xlabel('hop time')
        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "out.png"
        plt.show()
        plt.savefig(out_path, format='png', dpi=300, bbox_inches = 'tight')


        return Path(out_path)
