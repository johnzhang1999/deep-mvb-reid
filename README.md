# Torchreid MVB Fork Documentation

The MVB fork adds additional functionalities for re-ID research, experimentation, and visualization in addition to the Pytorch library Torchreid. It also contains a built-in interface for NUCTECH's Multi-View Baggage (MVB) dataset.

#### Notice
The MVB fork is still in active development so bugs and errors should be expected.

## Installation

You must be on Linux to install. 

```shell
$ cd deep-person-reid/
# yet to be updated...
$ pip install -r requirements.txt  # with pip
$ conda create -f environment.yml  # with conda
$ python setup.py develop
```

## MVB Dataset Preparation

The current setup assumes some root data folder `$DATA` with the structure:

``` shell
$DATA
└── MVB_train
    ├── Image
    │   ├── 0000_g_1.jpg
    │   ├── ...
    └── Info
        ├── sa_gallery.csv
        ├── sa_query.csv
        ├── sa_train.csv
        └── train.json
```

where `sa_{gallery,query,train}.csv` contain information about the gallery, query, and train identities in the form of `filename, pid, camid`.

The MVB data loader is located at `torchreid/data/datasets/image/mvb.py`.

## Usage

We suggest that you use the unified interface located at `scripts/main.py` to train your models. 

We provide some common flags that you'll probably often use here. For a full documentation of the interface, go to `scripts/default_parser.py` for descriptions and default values.

* Dataset root: `--root <path/to/dataset>`
* Sources (must have an interface): `-s` `--source` (for mvb, use `-s mvb`)
* Data type: `--app` (default: `image`)
* Loss: `--loss` (options: `softmax`, `triplet`) Note: combination of softmax and tirplet loss can be achieved by setting `--weight-t <triplet_weight>` and `--weight-x <softmax_weight>`. If you use triplet loss, `--train-sampler` must be set to `RandomIdentitySampler`
* Input dimension: `--height` and `--width`
* Label smoothing: `--label-smooth` (using label smoothing regularizer in cross entropy loss increases performance empirically)
* Model architecture: `-a` `--arch`. Full list see `torchreid/models/__init__.py`
* Optimizer config: `--optim <optimizer>` `--lr <learning_rate>` (suggest `adam` and `0.00065`)
* Lr scheduler: `--lr-scheduler <scheduler>` (default: `auto` which is ReduceLROnPlateau). `--patience`, `--gamma` (reduction factor), and `--min-lr` can be set. 
* Preprocessing: `--transforms random_crop random_flip random_erase`
* Training: `--max-epoch` `--fixbase-epoch` (used when finetuning) `--open-layers` (layer names to open while fixbase) `--load-weights` (pretrained weights) `--batch-size`  `--save-dir` (logging dir, where checkpoints, logs, visualizations, and tfboard events stay) `--use-avai-gpus`
* Resume: `--resume <path/to/checkpoint>`. Setting new `--lr` or lr_scheduler config flags will overwrite the checkpoint config. 
* Evaluation: `--start-eval` `--eval-freq` ` --ranks` (ranks you want to calculate)
* Visualization: `--viscam` `--viscam-num <k>` visualize the class activation map of k images in probe and gallery (respectively) after GAP (which layer can be changed in `torchreid/utils/cam.py`) CAMs stored in `$LOGDIR/viscam-<epoch>/`. `--visrank` `--visrank-topk` visualize the top k retrieved identities of each probe. Images stored in `$LOGDIR/visrank-<epoch>/`.
* Separate test/evaluation: add `--evaluate` and `--load-weights`. If only want to visualize CAM add `--viscam-only` (will raise `RuntimeError` after CAMs are saved)
* Rerank: `--rerank`
* Comet.ml: If you do not have your comet API key set up or do not want to log to comet, use `--no-comet` 
* Combine: `--combine-method`. Use combine method of `{none, mean}`. (`mean` means embeddings of different images of the same id will be averaged in retrieval to represent that identity).

A sample training script is as below:

```shell
python scripts/main.py --root ../../data --app image --loss softmax --label-smooth -s mvb -a osnet_custom --height 256 --width 256 --optim adam --lr 0.00065 --gamma 0.5 --max-epoch 180 --fixbase-epoch 5 --open-layers fc classifier --batch-size 50 --transforms random_crop random_flip random_erase --load-weights ~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth --save-dir log/osnet_custom-softmax-2048featdim --use-avai-gpus --start-eval 15 --eval-freq 5 --ranks 1 2 3 5 10 20
```

A sample testing script is as below:

```shell
python scripts/main.py --root ../../data --app image --loss softmax --label-smooth -s mvb -a osnet_x1_0 --height 256 --width 256 --batch-size 36 --load-weights log/osnet_x1_0-softmax/model.pth.tar-70 --evaluate --save-dir log/osnet_x1_0-softmax --use-avai-gpus --no-comet --rerank --viscam --viscam-num 20
```

### Logging

* Plain text log files (stdout) are stored in `$LOGDIR`.
* Tensorboard events are written to `scripts/runs/<stem($LOGDIR)>`.
* Metrics, hyperparameters, system metrics, stdout, cli flags, etc. are logged to Comet.ml unless `--no-comet` is set. Comet.ml config can be changed in `scripts/main.py`.

## Fork Contributor

Tianyu (John) Zhang @ NUCTECH, Bo (Hiroshi) Wu, Yicong (Michael) Mao



