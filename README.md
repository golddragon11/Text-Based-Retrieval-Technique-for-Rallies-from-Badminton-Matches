Text-Based Retrieval Technique for Rallies from Badminton Matches
===

## Table of Contents

[TOC]

## Environment Setup
* Setup conda environment with `conda install --file requirements.txt`.
* Download [TrackNetV3](https://github.com/qaz812345/TrackNetV3) and [extracting-video-features-ResNeXt](https://github.com/kaiqiangh/extracting-video-features-ResNeXt).

## Data Preparation

感謝易志偉教授實驗室提供「拍拍標記」資料集！

* Edit `directory` in [videoFeature.py](dataset/videoFeature.py), [preprocessing.py](dataset/preprocessing.py) to the dataset path.
* Run `python videoFeature.py` to have rally videos split into clips of single shots and have TrackNetV3 process each clip.
* Use [extracting-video-features-ResNeXt](https://github.com/kaiqiangh/extracting-video-features-ResNeXt) to extract video features.
* Run `python preprocessing.py` to generate pickle file.

## Training the Model
* Set hyperparameters by editing JSON files in [configs](configs)
* Train shot classification model by running
```shell
python train_stage1.py -c configs/training/config_stage1.json
```
* Train retrieval model by running
```shell
python train_stage2.py -c configs/training/config_stage2.json
```
