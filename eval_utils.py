from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
import string
import random
import shutil
import os
import sys
import misc.utils as utils
import subprocess
from six.moves import cPickle
import time

def extend_paragraph(sent_num,par_score):
    new_score = par_score.new(sum(sent_num)).zero_()
    m = 0
    for i,n in enumerate(sent_num):
        for j in range(n):
            new_score[m+j:m+j+1] = par_score[i]
        m+=n
    return new_score

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def language_eval_video(preds, model_id, split, diversity_dict, remove=False):
    import sys
    sys.path.append("movie_eval")
    results = []
    id = 1
    for pred in preds:
        sent = ' '.join([word for word in pred['caption'].split() if word != '<UNK>'])
        info = {'video_id': id, 'caption' : sent}
        results.append(info)
        id+=1
    if remove:
        model_id += id_generator() # to avoid processing and removing same ids
    split_ = split if split != "blind_test" else "blindtest"
    ref_path = os.path.join("data", "LSMDC16_annos_%s_someone.csv" % split_)
    with open(os.path.join('movie_eval', 'captions', 'caption_' + model_id + '.json'), 'w') as f:
        json.dump(results, f)
        f.close()
    eval_command = ["python","evaluate.py", "-s",'captions/caption_' + model_id + '.json',
                    "-o", 'results/result_' + model_id + '.json', "-r", ref_path, '--verbose']
    print(eval_command)
    subprocess.call(eval_command,cwd='movie_eval')

    # update and write with diversity statistics
    with open(os.path.join('movie_eval', 'results','result_' + model_id + '.json'),'r') as f:
        output = json.load(f)
        output.update(diversity_dict)
        f.close()
    with open(os.path.join('movie_eval', 'results','result_' + model_id + '.json'),'w') as f:
        json.dump(output,f)
        f.close()
    if remove: # remove for validation
        os.remove(os.path.join('movie_eval','captions','caption_' + model_id + '.json'))
        os.remove(os.path.join('movie_eval','results','result_' + model_id + '.json'))
    return output

def bigram(sent):
    return zip(sent.split(" ")[:-1], sent.split(" ")[1:])

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def diversity_meausures(predictions,div):
    vocab = {'gt': set(), 'gen': set()}
    sentences = {'gt' : {'total': [], 'unique': set()} , 'gen': {'total': [], 'unique': set()} }
    length = {'gt': [], 'gen': []}
    vocab_5 = {'gt' : set(), 'gen': set() }
    sentences_5 = {'gt' : {'total': [], 'unique': set()} , 'gen': {'total': [], 'unique': set()} }

    div_1 = {'gt' : [], 'gen': []}
    div_2 = {'gt' : [], 'gen': []}

    template = {'vocab_size' : {}, 'novel_sentences' : {} , 'sent_length': {}}

    for entry in predictions:
        for mode in ['gen', 'gt']:
            sent = entry['caption'] if mode == 'gen' else entry['gt']
            vocab[mode]|= set(sent.split())
            sentences[mode]['total'].append(sent)
            sentences[mode]['unique'].add(sent)
            length[mode].append(len(sent.split()))

    for mode in ['gen','gt']:
        template['vocab_size'][mode] = len(vocab[mode])
        template['novel_sentences'][mode] = round(len(sentences[mode]['unique']) / len(sentences[mode]['total']),3)
        template['sent_length'][mode] = np.mean(length[mode])

    for k in range(len(div['gen'])):
        for mode in ['gen','gt']:
            caption_list = div[mode][k]['captions'] # list of captions per image
            unigrams = [word for g in caption_list for word in g.split()]
            vocab_5[mode]|= set(unigrams)
            sentences_5[mode]['total'].extend(caption_list)
            sentences_5[mode]['unique']|= set(caption_list)
            div_1[mode].append(len(set(unigrams)) / len(unigrams))

            bigrams = [bg for g in caption_list for bg in bigram(g)]
            div_2[mode].append(len(set(bigrams)) / len(bigrams))

    if len(div_1['gen']) > 0: # diversity score for multiple captions
        for keys in ['vocab_size_5','novel_sentences_5','div_1','div_2']:
            template[keys] = {}
        for mode in ['gen','gt']:
            template['vocab_size_5'][mode] = len(vocab_5[mode])
            template['novel_sentences_5'][mode] = round(len(sentences_5[mode]['unique']) / len(sentences_5[mode]['total']),3)
            template['div_1'][mode] = round(np.mean(div_1[mode]),3)
            template['div_2'][mode] = round(np.mean(div_2[mode]),3)

    return template

def eval_split(gen_model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    dump_json = eval_kwargs.get('dump_json', 0)
    num_videos = eval_kwargs.get('num_videos', eval_kwargs.get('val_videos_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    use_context = eval_kwargs.get('use_context', 0)

    sample_max = eval_kwargs.get('sample_max', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    num_samples = eval_kwargs.get('num_samples', 1)
    num_captions = eval_kwargs.get('num_captions', 1)
    remove_caption = eval_kwargs.get('remove', 0) # usually remove captions in validation stage but not in test.
    seed = eval_kwargs.get('seed', 1234)


    model_id = eval_kwargs.get('id', eval_kwargs.get('val_id', ''))

    if split == 'val':
        model_id = 'val_' + model_id

    if sample_max:
        assert num_captions <= beam_size
    else:
        assert num_captions <= num_samples

    if use_context:
        gen_model.use_context()
    # Make sure in the evaluation mode
    gen_model.eval()

    loader.reset_iterator(split)

    n = 0
    losses = []
    predictions = []
    div = {'gt': [], 'gen': []}

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['img_feats'], data['box_feats'], data['labels'], data['masks'],
               data['mm_fc_feats'], data['mm_img_feats'], data['mm_box_feats'], data['mm_labels']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, img_feats, box_feats, labels, masks, \
        mm_fc_feats, mm_img_feats, mm_box_feats, mm_labels = tmp
        sent_num = data['sent_num']

        torch.manual_seed(seed)

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # calculate loss
            gen_seq = gen_model(fc_feats, img_feats, box_feats, labels)
            gen_seq = utils.align_seq(sent_num, gen_seq)
            loss = crit(gen_seq, utils.align_seq(sent_num, labels), utils.align_seq(sent_num, masks)).item()
            losses.append(loss)

            # use greedy max for inference
            if sample_max:
                eval_kwargs['sample_max'] = 1
                seq, _ = gen_model(fc_feats, img_feats, box_feats,
                               opt=eval_kwargs, mode='sample')

            # use sampling for inference
            else:
                sample_list = np.zeros((loader.batch_size, num_samples, loader.seq_length))
                context_list = np.zeros((loader.batch_size, num_samples, 512//4))
                seq_dummy = torch.zeros(loader.batch_size, 10, loader.seq_length).cuda()
                best_context = None
                for s in range(max(sent_num)):
                    prob_score_list = np.zeros((loader.batch_size, num_samples))
                    score_list = np.zeros((loader.batch_size, num_samples))
                    for i in range(num_samples):
                        fc_feats_s = fc_feats[:, s]
                        img_feats_s = img_feats[:, s]
                        box_feats_s = box_feats[:, s]
                        seq, logprobs, context = gen_model.sample_sequential(fc_feats_s, img_feats_s, box_feats_s,
                                                                             best_context, opt=eval_kwargs)
                        sample_list[:, i] = seq.cpu().numpy()
                        context_list[:, i] = context.squeeze(1)

                        prob_score = (torch.sum(logprobs, 1).cpu().numpy()) / np.count_nonzero(seq, axis=1)
                        prob_score_list[:, i] += prob_score
                        if score_list[:, i].sum() == 0:
                            score_list[:, i] += 0.5 * prob_score

                    # select the caption with highest score
                    inds = score_list.argsort(axis=1)[:, ::-1]
                    caption_list = torch.tensor(
                        sample_list[np.arange(loader.batch_size)[:, None], inds]).cuda().long()
                    best_context = torch.tensor(
                        context_list[np.arange(loader.batch_size)[:, None], inds][:, :1, :]).cuda().float()
                    best_seq = caption_list[:, 0, :]
                    seq_dummy[:, s] = best_seq

                # generated sequence
                seq = seq_dummy.long()

        seq = utils.align_seq(sent_num,seq)
        labels = utils.align_seq(sent_num,labels)
        mm_labels = utils.align_seq(sent_num, mm_labels)
        gt = utils.decode_sequence(loader.get_vocab(),labels[:,1:-1].data)
        mm = utils.decode_sequence(loader.get_vocab(), mm_labels[:,1:-1].data)
        seq = seq.data

        # print and store actual decoded sentence
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'video_id': data['infos'][k]['id'], 'caption': sent.encode('ascii', 'ignore').replace(" 's", "'s"),
                     'group_id' : data['infos'][k]['g_index'], 'gt' : gt[k].encode('ascii','ignore'), 'mm' : mm[k].encode('ascii','ignore'),
                     }

            predictions.append(entry)

            if verbose:
                print('video %s: caption: %s; gt: %s' %(entry['video_id'], entry['caption'], entry['gt']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_videos != -1:
            ix1 = min(ix1, num_videos)
        i = 0
        img_id = predictions[-1]['group_id']
        while i < (n-ix1):
            predictions.pop()
            cur_id = predictions[-1]['group_id']
            if cur_id != img_id:
                i+=1
                img_id = cur_id

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))


        if data['bounds']['wrapped']:
            break
        if num_videos >= 0 and n >= num_videos:
            break

    # Switch back to training mode
    gen_model.train()

    # calculate language metrics score
    gen_loss = np.mean(losses)
    lang_stats = None
    if lang_eval == 1:
        diversity_dict = diversity_meausures(predictions,div)
        diversity_dict.update({'loss': gen_loss})
        lang_stats = language_eval_video(predictions, model_id, split, diversity_dict, remove=remove_caption)
        print(lang_stats)

    if dump_json == 1:
        # dump the json
        json.dump(lang_stats, open('eval_results/' + model_id + '.json', 'w'))
        json.dump(predictions, open('vis/vis_' + model_id + '.json', 'w'))
        json.dump(div['gen'], open('vis/vis_n_' + model_id + '.json', 'w'))

    return gen_loss, predictions, lang_stats, div
