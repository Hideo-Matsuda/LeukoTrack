# Contents
- [Contents](#Contents)
- [This repository](#This-repository)
  - [Installation](##Installation)
  - [Data](##Data)
    - [Directory structure](###Directory-structure)
    - [detection file and ground truth file](###detection-file-and-ground-truth-file)
    - [sequence information file (seqinfo.ini)](###sequence-information-file-(seqinfo.ini))
- [Training](#Training)
- [Detection](#Detection)
- [Test (Tracking)](#test-(Tracking))
- [data augmentation](#data-augmentaion)

# This repository
This repository has been modified to apply [Tracktor (tracking without bells and whistles)](https://github.com/phil-bergmann/tracking_wo_bnw) to cell image data.

## Installation
Please refer to [here](https://github.com/phil-bergmann/tracking_wo_bnw) for environment construction. We have been tested under the environment of Python ３.７.10, pytorch １.７.1 and torchvision０.８.2.
1. Clone and enter this repository:
  ```
  git clone https://github.com/Hideo-Matsuda/LeukoTrack
  ```
2. Install packages for Python 3.7 in [virtualenv](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/):
  1. `pip install -r requirements.txt`
  2. Install PyTorch 1.7 and torchvision 0.8 from [here](https://pytorch.org/get-started/previous-versions/#v160).
  3. Install Tracktor: `pip install -e .`

## Data
1. Train data should be placed in `src/data/train/`.
2. Test data should be placed in `src/data/test/`.

The following is a description of the data format. Basically, it follows the MOT (multiple object tracking) competition data.

### Directory structure
The internal structure of data directory is as follows. The `group_name` is the name of the group (ex. group1, group2 etc.) which the data set was split for cross-validation. The `sample_name` is the sequence name (ex. sample01, sample02 etc.).
```
train or test
├── group_name(ex. group1)
│   ├── sample_name(ex. sample01)
│   │   ├── det
│   │   │   └── det.txt
│   │   ├── img1
│   │   │   └── 000001.png
│   │   │   └── 000002.png
│   │   ├── gt
│   │   │   └── gt.txt
│   │   ├── seqinfo.ini
│   ├── sample_name(ex. sample02)
├── group_name(ex. group2)
```

### detection file and ground truth file
The common format for the detection file (`det.txt`), the ground truth file (`gt.txt`), and the output file of the tracking results is as follows.one line corresponds to one bounding box (= detection) per frame and per target. Each line is separated by comma and lists the following information.

- frame 
  * int
- target id
  * int
  * In the detection file (since the correspondence between frames is not taken into account), all -1 is used.
- left x-coordinate of Bounding Box
  * int or float
- top y-coordinate of Bounding Box
  * int or float
- width of Bounding Box
  * int or float
- hight of Bounding Box
  * int or float
- Detection confidence
  * float
- class no.
  * In the detection file, all -1 is used.
- Visibility 
  * how is the target "visible" in the tracking results file. For example, lower values for targets that are not visible due to occlusion or being at the edges of the image.
  * In the detection file, all -1 is used.

For details, refer to Section 3.3 of [here](https://arxiv.org/abs/1603.00831).

### sequence information file (seqinfo.ini)
`seqinfo.ini` contains a total of seven pieces of information about `name`, `imgDir`, `frameRate`, `seqLength`, `imWidth`, `imHeight`, `imExt`.
The `name` is written in sample name(ex. sample01, sample02 ...), the `imgDir` is written in `img1` and the `frameRate` is written in seconds per frame.

# Training
1. Create config file
Create a yaml file that write the training setup　such as `train_cfgs/sample.yaml`.

2. Execution command
If you want to use the original dataset which have another data name structure, please modify `det/crest/crest_det_dataset`.
```
python src/det/train_detector.py -c train_cfgs/sample.yaml

```
- `-c`, `--cfg_file`: path to config file

After training, the model is saved as `{epoch}.pth` in the directory specified in the configuration file (`train_results/... `). If the directory already exists, the model is saved in the directory with the execution date and time appended at the end to distinguish it from other directories.

# Detection
Detects the target for all frames of the test sequence before testing (tracking).
1. Create config file
Create a yaml file that write the test sequence to be executed such as `test_group/sample.yaml`.
2. Execution command
```
python det/detect.py -d data/test -g test_group/sample.yaml -m train_results/sample_results/30.pth -o det_results/sample_results
```
- `-d`, `--dataroot`: path to test data directory
- `-g`, `--group`: config file with test sequences.(If omitted, it is executed for all sequences in the `dataroot`.)
- `-m`, `--model_path`: path to the train model (`.pth`).
- `-o`, `--outdir`: path to the directory where the detection results will be saved.

After the detection, the detection results of MOT format are saved in the directory which specified by `outdir`. If the directory already exists, the directory with the execution date and time appended to the end of the file will be used as the destination directory to distinguish it.

# test (Tracking)
1. Download ReID model
The ReID model trained on generic objects is in `src/reid_model/model-last.pth.tar`. This model was downloaded from [here](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v5.zip) as per the instructions in [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). If you want to use another ReID model, put in `src/reid_model/`.
2. Data preparation
Copy the detection results for each sequence in `det/det_results/{group_name}` to `det/det.txt` in the data directory (`data/test/{group_name}/{sample_name}`). 
3. Create config file
Create a yaml file that write the training setup such as `test_cfgs/sample.yaml`.　The `dataset` in the this config file (`test_cfgs/sample.yaml`) specifies the dataset to track.
```
Format: 
biodata_{group}
```
- `group`
  - `group1`, `group2`, ... etc. : group name (only test sequence in group)
  - `all`: test sequence in group + train sequence in group

If you want to apply the another structure format, modify code by imitating lines 50-52 in `src/track/tracktor/datasets/factory.py`

4. Specify dataset to track
Change the test data names by group (See `class BiodataWrapper` in `src/track/tracktor/datasets/mot_wrapper.py`) to correspond to the `dataset` in the config file.

5. Execution command
If you want to use the original dataset which have another data name structure, please modify `det/crest/crest_det_dataset`.
```
python src/scripts/test_tracktor.py -c test_cfgs/sample.yaml
```
- `-c`, `--cfg_file`: path to config file.

After the tracking, the tracking results are saved in the directory which specified by `output/tracker/{module_name}/{name}`. `{module_name}` and `{name}` are in the config file.
If the directory already exists, the directory with the execution date and time appended to the end of the file will be used as the destination directory to distinguish it.

# data augmentaion
1. Change the parameters in the `src/data_augmentation_by_image_processing.py` and run.
```
python src/data_augmentation_by_image_processing.py
```
2. Change the parameters in the `src/create_pseudo_labels.py` and run.
```
python src/create_pseudo_labels.py
```
=======
# LeukoTrack
Leukocyte tracking by CNN
