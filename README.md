# W-RIZZ
Repository for W-RIZZ (RA-L 2024)


W-RIZZ: A Weakly-Supervised Framework for Relative Traversability Estimation in Mobile Robotics

Andre Schreiber, Arun N. Sivakumar, Peter Du, Mateus V. Gasparino, Girish Chowdhary, and Katherine Driggs-Campbell

*Abstract*: Successful deployment of mobile robots in unstructured domains requires an understanding of the environment and terrain to avoid hazardous areas, getting stuck, and colliding with obstacles. Traversability estimation--which predicts where in the environment a robot can travel--is one prominent approach that tackles this problem. Existing geometric methods may ignore important semantic considerations, while semantic segmentation approaches involve a tedious labeling process. Recent self-supervised methods reduce labeling tedium, but require additional data or models and tend to struggle to explicitly label untraversable areas. To address these limitations, we introduce a weakly-supervised method for relative traversability estimation. Our method involves manually annotating the relative traversability of a small number of point pairs, which significantly reduces labeling effort compared to traditional segmentation-based methods and avoids the limitations of self-supervised methods. We further improve the performance of our method through a novel cross-image labeling strategy and loss function. We demonstrate the viability and performance of our method through deployment on a mobile robot in outdoor environments.

## Data setup

Any command line instructions provided in this README assume that you have the following folders (in the same directory as this README):
* ``experiments/``
* ``data/``
* ``pretrained_weights/``

To setup the data for training, go into the ``data/`` folder and run the following:
```
curl -L  https://uofi.box.com/shared/static/wryubgcx24y0i3bt8n6cbkqpmtg8pe02 --output wayfast_dataset.zip
unzip wayfast_dataset.zip
rm wayfast_dataset.zip
mv data wayfast
```
which will download the WayFAST dataset and rename it to be in a folder ``wayfast/`` within ``data/``.

Then, download the annotation .csv files from https://uofi.box.com/s/o8fxnvblf0lmx1ph2dmt5v1yq8lkkqk4, and move them into the folder ``data/wayfast``

If you wish to initialize training with weights from a RUGD training, move back to the root directory and download the .pth weights file (from pretraining on RUGD) from https://uofi.box.com/s/st2c2csduq47xxfc7dfclwbmqtfw5de2. Move the weights file into the ``pretrained_weights/`` folder.

## Running a training

Then, to run a training, use:
```
python train.py \
    --seed 123 \
    --train_folder_path ./data/wayfast/rgb \
    --train_csv_path ./data/wayfast/train_labels.csv \
    --valid_folder_path ./data/wayfast/rgb \
    --valid_csv_path ./data/wayfast/valid_labels.csv \
    --batch_size 8 \
    --num_workers 8 \
    --model travnetup3nnrgb \
    --use_mean_teacher \
    --epochs 30 \
    --rebalance \
    --loss lrizz \
    --experiment experiments/my_experiment
```

To use pretrained weights for initialization, do
```
python train.py \
    --seed 123 \
    --train_folder_path ./data/wayfast/rgb \
    --train_csv_path ./data/wayfast/train_labels.csv \
    --valid_folder_path ./data/wayfast/rgb \
    --valid_csv_path ./data/wayfast/valid_labels.csv \
    --batch_size 8 \
    --num_workers 8 \
    --model travnetup3nnrgb \
    --use_mean_teacher \
    --epochs 30 \
    --rebalance \
    --loss lrizz \
    --pretrained_weights pretrained_weights/TravNetUp3NN_RUGD_Pretrain.pth \
    --experiment experiments/my_experiment
```

## Evaluation

Evaluation of an experiment can be performed using:
```
python test.py \
    --seed 123 \
    --experiment experiments/my_experiment \
    --output experiments/my_experiment/results_test.json
```

## Model weights

Model weights for the W-RIZZ model (with L-RIZZ, pretraining on RUGD, and mean teacher) can be found here: https://uofi.box.com/s/s73bnggeo48o8iyzem4c5w6aduc208h2.

## Environment setup
You can set up the environment using the provided ``environment.yml`` conda environment file (``conda env create -f environment.yml``).

Alternatively, the following sequence of commands installs the environment:
```
conda create -n wrizz python=3.10
conda activate wrizz
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install anaconda::pandas
conda install conda-forge::matplotlib
conda install anaconda::scikit-image
conda install anaconda::scipy
conda install conda-forge::tqdm
pip install torch-ema
```

## Questions
Code has been tested on Ubuntu (18.04) with Python 3.10 and PyTorch 2.0.1 (CUDA 11.7).

If you have any questions or concerns, feel free to raise an issue or contact the author (andrems2@illinois.edu).