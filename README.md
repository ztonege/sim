# Shift Invariant Module
Project Page: https://ztonege.github.io/sim-project-page/

## Setup Development Environment

1. Create environment using `conda env create environment.yml`
2. Install pre-commit hooks using `pre-commit install`
3. Activate environment before running anything using `conda activate aam`

## Downloading and Preprocessing Dataset

Before running train or evaluation, you will need to download and preprocess the dataset.

1. Cifar10

   1. Preprocess the dataset by running

   ```console
   python ./preprocess/write_cifar.py --config-file configs/cifar_config.yaml
   ```

2. Imagenet

   1. Download dataset using `kaggle competitions download -c imagenet-object-localization-challenge`
      1. You can download kaggle cli using `conda install -c conda-forge kaggle`
      2. Create a kaggle api key and save it to `~/.kaggle/kaggle.json` (follow instructions [here](https://www.kaggle.com/docs/api))
      3. Before downloading, you will also need to accept the competition rules by going to [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/) and clicking on "Late Submission".
   2. After downloading, download and run this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) in `<IMAGENET_DIR>/val` to move validation images to labeled subfolders.
   3. Preprocess the data by running

   ```console
   export IMAGENET_DIR=</path/to/pytorch/format/imagenet/directory>
   export WRITE_DIR=data

   ./preprocess/write_imagenet.sh 400 0.1 90
   ```

## Run Training

### Classification

Currently 2 datasets are supported, ImageNet & Cifar10. You can train a resnet18 from scratch by running `python main.py --config <config path>`. For example, if you want to train a vanilla resnet18 with using cifar10, run

```console
python main.py --config configs/cifar_config.yaml
```

## Results

### Imagenet

#### ResNet

| Model    | Consistency | Top 1       | Top 5       |
| -------- | ----------- | ----------- | ----------- |
| Original | 83.012      | 65.3699     | 87.0039     |
| AACNN    | 87.834      | **69.1259** | **89.0500** |
| Group    | **90.422**  | 68.7780     | 88.7399     |
| 3D       | 90.200      | 68.5620     | 88.4079     |

# References

A lot of code is copied from

1. https://github.com/adobe/antialiased-cnns
2. https://github.com/libffcv/ffcv-imagenet
3. https://github.com/libffcv/ffcv
4. https://github.com/fastai/imagenet-fast/
