# Baseline for Large Scale Movie Description Challenge

This is a sample baseline for LSMDC19 Task 1 "Multi Sentence Video Description". 

This code is written in pytorch and is based on [adv-inf](https://github.com/jamespark3922/adv-inf) and [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) repositories.

## Requirements
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)  
PyTorch 0.4       
[movie_eval](https://github.com/jamespark3922/movie_eval) (for evaluation but downloaded recursively)  
[coco-caption](https://github.com/tylin/coco-caption) (located in movie_eval; run `./get_stanford_models.sh` to set up SPICE evaluation)

Clone the repository recursively to get the submodules.

```git clone --recursive https://github.com/jamespark3922/lsmdc```

## Download LSMDC captions and preprocess them
Download caption csv files from the [website](https://sites.google.com/site/describingmovies/download?authuser=0). 
Then, run the following command to preprocess the captions.
```bash
python prepro_vocab.py --input_path $CSV_PATH
```
This will result in two files, `LSMDC16_info_someone.json` and `LSMDC16_labels_someone.h5`, which will be used as inputs for training and testing.

Then, move the csv files to `movie_eval/data` to be used as reference for evaluation.
```bash
mkdir movie_eval/data
mv *.csv movie_eval/data
```

## Precomputed Features
The website shares precomputed visual features in the download [page](https://sites.google.com/site/describingmovies/download?authuser=0).
- **i3d (3.3GB)** 
- **resnet152 (6.8GB)**

After downloading them all, unzip them to your preferred feature directory.

## Training
```bash
python train.py --input_json LSMDC16_info_someone.json --input_fc_dir feats/i3d/ --input_img_dir feats/resnet152/ --input_label_h5 LSMDC16_labels_someone.h5 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --val_videos_use -1 --losses_print_every 10 --batch_size 16 
```
The training scheme follows the format in [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) repository.

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = save/).   
We save the model every epoch, along with the best-performing checkpoint on validation and the latest checkpoint.  
To resume training, you can specify `--g_start_from` with corresponding `--g_start_epoch` (default = 'latest') option to be the path saving `infos.pkl` and `model_$G_START_EPOCH.pth` (usually you could just set `--g_start_from` and `--checkpoint_path` to be the same).  

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the coco-caption code.

You can see more details in `opts.py`

**Context**: The generator model uses the hidden state of previous sentence as "context", starting at epoch `--g_context_epoch` (default = after 10 epochs); before then, the context feature is zero-padded.

## Pretrained Model

We share pretrained models here.

The results compared to previous challenge submissions in public test set are: 

## Evaluation
After training is done, one can evaluate the captions in the public test set.

The normal inference using greedy max or beamsearch can be run with the following command:
```angular2html
python eval.py --g_model_path save/gen_best.pth --infos_path save/infos.pkl  --sample_max 1 --id $id --beam_size $beam_size
```
You can also do random sampling by setting `--sample_max 0` and specifying `--top_num n` option. (n > 1 performs top-k sampling, 0 < n < 1 performs nucleus sampling)

If you are using the pretrained model, then you will need to update the path to your input files.
```angular2html
python eval.py --g_model_path save/gen_best.pth --infos_path save/infos.pkl  --sample_max 1 --id $id --beam_size $beam_size --input_json LSMDC16_info_someone.json --input_fc_dir feats/i3d/ --input_img_dir feats/resnet152/ --input_label_h5 LSMDC16_labels_someone.h5
```

To run on the blind test split, simply add `--split blind_test` option.

The captions will be saved in `movie_eval/captions/caption_$id.json` and results in `movie_eval/results/result_$id.json`

## Reference

```
@article{rohrbach2015movie,
  title= A Dataset for Movie Description,
  author={Rohrbach, Anna and Rohrbach, Marchus and Tandon, and Schiele, Bernt},
  jorunal={CVPR 2015},
  year={2015}
}

@article{park2019advinf,
  title= Adversarial Inference for Multi-Sentence Video Descriptions,
  author={Park, Jae Sung and Rohrbach, Marcus and Darrell, Trevor and Rohrbach, Anna},
  jorunal={CVPR 2019},
  year={2019}
}
```
