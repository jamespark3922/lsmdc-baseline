from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

def zero_pad(features,n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack((features,np.zeros((n_feat - features.shape[0], features.shape[1]))))
    return features

# https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list
def random_derangement(n):
    if n == 0 or n == 1:
        return n
    while True:
        v = range(n)
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_wtoi(self):
        return self.word_to_ix

    def get_seq_length(self):
        return self.seq_length

    def get_sos_token(self):
        return self.word_to_ix['<sos>']

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        self.input_fc_dir = self.opt.input_fc_dir
        self.use_video = getattr(opt, 'use_video', 0)
        # use other features
        self.use_img = getattr(opt, 'use_img', 0)
        if self.use_img:
            self.input_img_dir = self.opt.input_img_dir
        self.use_box = getattr(opt, 'use_box', 0)
        if self.use_box:
            self.input_box_dir = self.opt.input_box_dir
        self.feat_type = opt.feat_type
        self.nbox = 3

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.word_to_ix = self.info['word_to_ix']
        self.groups = self.info['groups']
        self.movie_dict = self.info['movie_ids']
        self.seq_length = self.info['max_seq_length']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.max_sent_num = self.opt.max_sent_num
        print('max sequence length in data is', self.seq_length)

        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r')
        self.labels = self.h5_label_file['labels'].value

        self.max_seg = opt.max_seg
        self.mean = opt.use_mean
        self.negatives = opt.negatives

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'blind_test': []}
        self.split_size = {'train': 0, 'val': 0, 'test': 0, 'blind_test': 0}
        self.ix_split = {}
        for j, group in enumerate(self.groups):
            split = group['split']
            self.split_ix[split].append(j)
            self.split_size[split]+=1
            self.ix_split[j] = split

        print('assigned %d videos to split train' % len(self.split_ix['train']))
        print('assigned %d videos to split val' % len(self.split_ix['val']))
        print('assigned %d videos to split test' % len(self.split_ix['test']))
        print('assigned %d videos to split blind_test' % len(self.split_ix['blind_test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'blind_test': 0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)


    # mean pool the features across $max_seg segments
    def meanpool_segments(self, features):
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            nps = int(np.floor(features.shape[0] // self.max_seg))  # numbers per segment
            for i in range(self.max_seg):
                if i != self.max_seg - 1:
                    segment = features[nps * i:nps * (i + 1)]
                else:
                    segment = features[nps * i:]
                segment = segment.mean(axis=0)
                tmp_feat.append(segment)
            features = np.array(tmp_feat)
        else:
            # 0 pad frames
            features = zero_pad(features, self.max_seg)
        return features

    def get_sent_num(self, index):
        return len(self.groups[index]['videos'])

    def get_label_batch(self, index):
        v_idx = self.groups[index]['videos']
        return self.labels[v_idx]

    def get_seg_batch(self, index, mode):
        v_idx = self.groups[index]['videos']
        sent_num = len(v_idx)
        assert sent_num > 0, 'data should have at least one caption'
        features = []
        for id in v_idx:
            movie = self.info['videos'][id]['movie']
            clip = self.info['videos'][id]['clip']
            if mode == 'video':
                if not self.use_video:
                    return None
                npy_dir = [self.input_fc_dir, movie, clip + '.npy']
            elif mode == 'img':
                if not self.use_img:
                    return None
                npy_dir = [self.input_img_dir, movie, clip + '.npy']
            else:
                raise AttributeError("mode %s not found" % mode)
            tmp_fc = self.meanpool_segments(np.load(os.path.join(*npy_dir)))
            features.append(tmp_fc)
        return np.array(features)

    def get_box_batch(self, index):
        if not self.use_box:
            return None
        v_idx = self.video_id[index]
        id = self.info['videos'][v_idx]['id']
        sent_num = self.sent_num[index]
        assert sent_num > 0, 'data should have at least one caption'
        box_features = []
        split = self.ix_split[index]
        if split == 'val':
            split = 'val2'
        elif split == 'test':
            split = 'val1'
        dir = os.path.join(self.input_box_dir,split)
        feats = np.load(os.path.join(dir,id + '.npy'))
        assert feats.shape[0] >= 3 * sent_num, 'weird feature for %s' % id
        for i in range(sent_num):
            box_features.append(feats[i*self.nbox:(i+1)*self.nbox])
        return box_features

    def set_negatives(self,mode):
        self.negatives = mode

    def build_glove(self,glove_path):
        pre, ext = os.path.splitext(glove_path)
        if ext != '.npy':
            embeddings_index = dict()
            f = open(glove_path)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            print('done processing this')
            embedding_matrix = np.zeros((self.vocab_size, 300))
            for word, index in self.word_to_ix.items():
                if word not in embeddings_index:
                    embedding_matrix[index] = np.random.normal(scale=0.6, size=(300,))
                else:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[index] = embedding_vector
            np.save(pre + '.npy', embedding_matrix)
            return embedding_matrix
        else:
            return np.load(glove_path)

    # Each batch is a video with multiple clips/sentences
    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # inputs for training
        sent_num_batch = np.zeros(batch_size, dtype='int')
        fc_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.fc_feat_size], dtype = 'float32')
        img_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.img_feat_size], dtype = 'float32')
        box_batch = np.zeros([batch_size, self.max_sent_num, self.nbox, self.opt.box_feat_size], dtype = 'float32')
        label_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length), dtype='int')
        mask_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length), dtype='float32')

        # negative inputs for discriminator
        mm_fc_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.fc_feat_size], dtype = 'float32')
        mm_img_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.img_feat_size], dtype = 'float32')
        mm_box_batch = np.zeros([batch_size, self.max_sent_num, self.nbox, self.opt.box_feat_size], dtype = 'float32')
        mm_label_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length), dtype = 'int')
        mm_mask_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length), dtype='float32')

        wrapped = False
        infos = []
        for i in range(batch_size):
            # fetch visual features
            tmp_fcs, ix, tmp_wrapped = self._prefetch_process[split].get()
            sent_num = self.get_sent_num(ix)
            fc_batch[i,:sent_num] = tmp_fcs[0]
            img_batch[i,:sent_num] = tmp_fcs[1]
            box_batch[i,:sent_num] = tmp_fcs[2]
            sent_num_batch[i] = sent_num
            label_batch[i,:sent_num] = self.get_label_batch(ix)

            # get visually mismatched (mm) captions and features for discriminator
            if self.negatives == 'video' and sent_num > 1:
                dl = random_derangement(sent_num)
                mm_label_batch[i, :sent_num] = label_batch[i,dl]
                mm_fc_batch[i, :sent_num] = fc_batch[i,dl]
                mm_img_batch[i, :sent_num] = img_batch[i,dl]
                mm_box_batch[i, :sent_num] = box_batch[i,dl]

            elif self.negatives in ['movie', 'video']:  # get caption from the same movie (hard negatives)
                movie = self.groups[ix]['movie']
                m = 0
                while True:
                    m_ix = random.choice(self.movie_dict[movie]) # get random group index in the movie
                    n = min(sent_num - m, self.get_sent_num(m_ix))
                    if self.groups[m_ix] != ix:  # avoid getting the gt pair
                        mm_label_batch[i, m:m + n] = self.get_label_batch(m_ix)[:n]
                        mm_fc_batch[i, m:m + n] = self.get_seg_batch(m_ix, "video")[:n] if self.use_video else None
                        mm_img_batch[i, m:m + n] = self.get_seg_batch(m_ix, "img")[:n] if self.use_img else None
                        mm_box_batch[i, m:m + n] = self.get_box_batch(m_ix)[:n] if self.use_box else None
                        m += n
                    if m >= sent_num:
                        break
            else:  # get random caption (random negatives)
                while True:
                    mmix = random.randint(0, len(self.split_ix[split]) - 1)
                    if self.groups[mmix] != ix and sent_num <= self.get_sent_num(mmix):  # avoid getting the gt pair
                        mm_label_batch[i, :sent_num, :self.seq_length] = self.get_label_batch(mmix)[:sent_num]
                        mm_fc_batch[i, :sent_num] = self.get_seg_batch(mmix, "video")[:sent_num] if self.use_video else None
                        mm_img_batch[i, :sent_num] = self.get_seg_batch(mmix, "img")[:sent_num] if self.use_img else None
                        mm_box_batch[i, :sent_num] = self.get_box_batch(mmix)[:sent_num] if self.use_box else None
                        break

            if tmp_wrapped:
                wrapped = True

            for v_ix in self.groups[ix]['videos']:
                info_dict = {}
                info_dict['index'] = v_ix
                info_dict['g_index'] = ix
                info_dict['id'] = self.info['videos'][v_ix]['clip']
                infos.append(info_dict)

            # generate mask
            nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch[i])))
            for ix, row in enumerate(mask_batch[i]):
                if ix < sent_num:
                    row[:nonzeros[ix]] = 1

            # generate mismatch mask
            nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, mm_label_batch[i])))
            for ix, row in enumerate(mm_mask_batch[i]):
                if ix < sent_num:
                    row[:nonzeros[ix]] = 1

        data = {}

        data['fc_feats'] = fc_batch
        data['img_feats'] = img_batch
        data['box_feats'] = box_batch
        data['labels'] = label_batch
        data['masks'] = mask_batch

        data['mm_fc_feats'] = mm_fc_batch
        data['mm_img_feats'] = mm_img_batch
        data['mm_box_feats'] = mm_box_batch
        data['mm_labels'] = mm_label_batch
        data['mm_masks'] = mm_mask_batch

        data['sent_num'] = sent_num_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        return [self.get_seg_batch(index,"video"), self.get_seg_batch(index,"img"),
                self.get_box_batch(index)], index

    def __len__(self):
        return len(self.info['videos'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]