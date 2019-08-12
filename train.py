from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import time
import json
import os

from six.moves import cPickle

import opts
import models
from dataloader import *
from train_utils import *
from eval_utils import eval_split
import misc.utils as utils

import gc
from models.MultiModalGenerator import MultiModalGenerator

# try:
#     import tensorboardX as tb
# except ImportError:
#     print("tensorboardX is not installed")
#     tb = None

# There seems to be cpu memory leak in lstm?
# https://github.com/pytorch/pytorch/issues/3665
torch.backends.cudnn.enabled = False

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    if not os.path.exists(opt.checkpoint_path):
        os.mkdirs(opt.checkpoint_path)

    with open(os.path.join(opt.checkpoint_path,'config.json'),'w') as f:
        json.dump(vars(opt),f)

    # Load iterators
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.sos_token = loader.get_sos_token()
    opt.seq_length = loader.seq_length
    opt.video = 1
    if opt.glove is not None:
        opt.glove_npy = loader.build_glove(opt.glove)
    else:
        opt.glove_npy = None

    # set up models
    gen_model = MultiModalGenerator(opt)
    gen_model = gen_model.cuda()
    gen_model.train()
    gen_optimizer = utils.build_optimizer(gen_model.parameters(), opt)

    # loss functions
    crit = utils.LanguageModelCriterion()
    gan_crit = nn.BCELoss().cuda()

    # keep track of iteration
    g_iter = 0
    g_epoch = 0
    dis_flag = False
    joint_flag = False
    update_lr_flag = True

    # Load from checkpoint path
    infos = {'opt': opt}
    histories = {}
    infos['vocab'] = loader.get_vocab()
    if opt.g_start_from is not None:
        # Open old infos and check if models are compatible
        with open(os.path.join(opt.g_start_from, 'infos.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # Load train/val histories
        with open(os.path.join(opt.g_start_from, 'histories.pkl')) as f:
            histories = cPickle.load(f)

        # Load generator
        g_start_epoch = opt.g_start_epoch
        g_model_path = os.path.join(opt.g_start_from, "gen_%s.pth" % g_start_epoch)
        g_optimizer_path = os.path.join(opt.g_start_from, "gen_optimizer_%s.pth" % g_start_epoch)
        assert os.path.isfile(g_model_path) and os.path.isfile(g_optimizer_path)
        gen_model.load_state_dict(torch.load(g_model_path))
        gen_optimizer.load_state_dict(torch.load(g_optimizer_path))
        if "latest" not in g_start_epoch and "best" != g_start_epoch:
            g_epoch = int(g_start_epoch) + 1
            g_iter = (g_epoch) * loader.split_size['train'] // opt.batch_size
        elif g_start_epoch == "best":
            g_epoch = infos['g_epoch_' + g_start_epoch] + 1
            g_iter = (g_epoch) * loader.split_size['train'] // opt.batch_size
        else:
            g_epoch = infos['g_epoch_' + g_start_epoch] + 1
            g_iter = infos['g_iter_' + g_start_epoch]
        print('loaded %s (epoch: %d iter: %d)' % (g_model_path, g_epoch, g_iter))

    infos['opt'] = opt
    loader.iterators = infos.get('g_iterators', loader.iterators)

    # misc
    best_val_score = infos.get('g_best_score', None)
    opt.seq_length = loader.seq_length
    opt.video = 1
    g_val_result_history = histories.get('g_val_result_history', {})
    g_loss_history = histories.get('g_loss_history', {})

    """ START TRAINING """
    while True:
        gc.collect()
        # set every epoch
        if update_lr_flag:
            # Assign the learning rate for generator
            if g_epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (g_epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(gen_optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if g_epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (g_epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                gen_model.ss_prob = opt.ss_prob

            # Start using previous sentence as context for generator (default: 10 epoch)
            if opt.g_context_epoch >= 0 and g_epoch >= opt.g_context_epoch:
                gen_model.use_context()

            update_lr_flag = False

        """ TRAIN GENERATOR """
        if not dis_flag:
            gen_model.train()

            # train generator
            start = time.time()

            gen_loss, wrapped, sent_num = train_generator(gen_model, gen_optimizer, crit, loader)
            end = time.time()

            # Print Info
            if g_iter % opt.losses_print_every == 0:
                print("g_iter {} (g_epoch {}), gen_loss = {:.3f}, time/batch = {:.3f}, num_sent = {} {}" \
                    .format(g_iter, g_epoch, gen_loss, end - start,sum(sent_num),sent_num))

            # Log Losses
            if g_iter % opt.losses_log_every == 0:
                g_loss = gen_loss
                g_loss_history[g_iter] = {'g_loss': g_loss, 'g_epoch': g_epoch}

            # Update the iteration
            g_iter += 1

            #########################
            # Evaluate & Save Model #
            #########################
            if wrapped:
                # evaluate model on dev set
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json,
                               'sample_max' : 1,
                               'language_eval': opt.language_eval,
                               'id' : opt.val_id,
                               'val_videos_use' : opt.val_videos_use,
                               'remove' : 1} # remove generated caption
                # eval_kwargs.update(vars(opt))

                val_loss, predictions, lang_stats, _ = eval_split(gen_model, crit, loader, eval_kwargs=eval_kwargs)

                if opt.language_eval == 1:
                    current_score = lang_stats['METEOR']
                else:
                    current_score = - val_loss
                g_val_result_history[g_epoch] = {'g_loss': val_loss, 'g_score': current_score, 'lang_stats': lang_stats}

                # Save the best generator model
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'gen_best.pth')
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_best.pth'))
                    infos['g_epoch_best'] = g_epoch
                    infos['g_iter_best'] = g_iter
                    infos['g_best_score'] = best_val_score
                    torch.save(gen_model.state_dict(), checkpoint_path)
                    print("best generator saved to {}".format(checkpoint_path))

                # Dump miscalleous informations and save
                infos['g_epoch_latest'] = g_epoch
                infos['g_iter_latest'] = g_iter
                infos['g_iterators'] = loader.iterators
                histories['g_val_result_history'] = g_val_result_history
                histories['g_loss_history'] = g_loss_history
                with open(os.path.join(opt.checkpoint_path, 'infos.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                # save the latest model
                if opt.save_checkpoint_every > 0 and g_epoch % opt.save_checkpoint_every == 0:
                    torch.save(gen_model.state_dict(), os.path.join(opt.checkpoint_path, 'gen_%d.pth'% g_epoch))
                    torch.save(gen_model.state_dict(), os.path.join(opt.checkpoint_path, 'gen_latest.pth'))
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_%d.pth'% g_epoch))
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_latest.pth'))
                    print("generator model saved to {} at epoch {}".format(opt.checkpoint_path, g_epoch))

                # update epoch and lr
                g_epoch += 1
                update_lr_flag = True

if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)
