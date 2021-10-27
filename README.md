# Toward interpretable polyphonic sound event detection with attention maps based on local prototypes

This repository includes the code for running the experiments reported in

Zinemanas, P.; Rocamora, M.; Fonseca, E.; Font, F.; Serra, X. [Toward interpretable polyphonic sound event detection with attention maps based on local prototypes](http://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Zinemanas_22.pdf). Proceedings of the Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021). Barcelona, Spain.

## Installation instructions
This repository uses DCASE-models and therefore please follow the recomendations from this library: 

We recommend to install DCASE-models in a dedicated virtual environment. For instance, using [anaconda](https://www.anaconda.com/):
```
conda create -n apnet python=3.6
conda activate apnet
```
For GPU support:
```
conda install cudatoolkit cudnn
```
DCASE-models uses [SoX](http://sox.sourceforge.net/) for functions related to the datasets. You can install it in your conda environment by:
```
conda install -c conda-forge sox
```
Before installing the library, you must install only one of the Tensorflow variants: CPU-only or GPU.
``` 
pip install "tensorflow<1.14" # for CPU-only version
pip install "tensorflow-gpu<1.14" # for GPU version
```

Now please install DCASE-models:
``` 
pip install "DCASE-models==0.2.0-rc0"
```

Now you can clone and use this repository:
``` 
git clone https://github.com/pzinemanas/attprotos.git
cd attprotos
```

## Usage

### Download datasets
``` 
cd experiments
python download_datasets.py -d URBAN_SED

```
### Train models
``` 
python train.py -m AttProtos -d URBAN_SED -f MelSpectrogram -fold test
``` 

### Evaluate models
``` 
python evaluate.py -m AttProtos -d URBAN_SED -f MelSpectrogram
``` 
