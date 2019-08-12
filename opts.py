import argparse
#
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str,
                    help='path to the json file containing additional info and vocab (img/video)')
    parser.add_argument('--input_fc_dir', type=str,
                        help='path to the directory containing the preprocessed fc video features')
    parser.add_argument('--input_img_dir', type=str,
                        help='path to the directory containing the image features')
    parser.add_argument('--input_box_dir', type=str,
                    help='path to the directory containing the boxes of att img feats (img)')
    parser.add_argument('--input_label_h5', type=str,
                    help='path to the h5file containing the preprocessed dataset (img/video)')

    parser.add_argument('--g_start_from', type=str, default=None,
                     help="""skip pre training step and continue training from saved generator model at this path.
                          'infos_{id}.pkl'         : configuration;
                          'gen_optimizer_{epoch}.pth'     : optimizer;
                          'gen_{epoch}.pth'         : model
                     """)
    parser.add_argument('--g_start_epoch', type=str, default="latest",
                     help="""start training generator at epoch (int, latest, latest_ce, latest_scst)
                     """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="video",
                    help='fc, show_tell, adaatt, topdown, s2vt, paragraph show_attend_tell, all_img, att2in, att2in2, att2all2,  stackatt, denseatt')
    parser.add_argument('--g_context_epoch', type=int, default=10,
                        help='epoch to start incorporating context for generator (-1 = dont use context)')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--video_encoding_size', type=int, default=256,
                        help='the encoding size of each frame of visual features.')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--feat_type', type=str, default='i3d',
                        help='feat type for video (i3d, c3d, resnext101-64f)')
    parser.add_argument('--fc_feat_size', type=int, default=1024,
                    help='1024 for i3d, 2048 for resnet, 4096 for vgg (img) \
                          500  for c3d,    8192 for r3d (video')
    parser.add_argument('--img_feat_size', type=int, default=2048,
                        help='img feat size')
    parser.add_argument('--box_feat_size', type=int, default=15461,
                        help='box feat size')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')
    parser.add_argument('--glove', type=str, default=None,
                        help='text or npy containing glove vector associated with word_idx labels. \
                                 builds a npy file in the same directory if text file is given')

    # input settings
    parser.add_argument('--use_video', type=int, default=1,
                        help='use video features (c3d/resnext101-64f) specified in input_fc_dir')
    parser.add_argument('--use_img', type=int, default=1,
                        help='use resnet features specified in input_img_dir')
    parser.add_argument('--use_box', type=int, default=0,
                        help='use bottomup features sepcified in input_box_dir')
    parser.add_argument('--max_seg', type=int, default=3,
                        help='number of segments to divide the temporal visual features')
    parser.add_argument('--max_sent_num', type=int, default=5,
                        help='max number of sentences per group (LSMDC has a group of 5 clips)')
    parser.add_argument('--use_mean', type=int, default=0)

    # Optimization: General
    parser.add_argument('--g_pre_nepoch', type=int, default=80,
                    help='number of epochs to pre-train generator with cross entropy')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_id', type=str, default='',
                        help='id to use to save captions for validation')
    parser.add_argument('--val_videos_use', type=int, default=-1,
                    help='how many videos to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--losses_print_every', type=int, default=10,
                    help='How often do we want to print losses? (0 = disable)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1,
                    help='how often to save a model checkpoint in iterations? the code already saves checkpoint every epoch (0 = dont save; 1 = every epoch)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.save_checkpoint_every >= 0, "saving checkpoint at every $epoch should be non-negative"

    return args
