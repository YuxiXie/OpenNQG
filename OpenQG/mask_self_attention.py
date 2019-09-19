import random
import argparse
from tqdm import tqdm
import numpy as np

import torch


def load_file(filename):
    data = torch.load(filename)
    data = {'train': data['train']['src'], 'valid': data['valid']['src']}
    return data


def dump_file(data, filename):
    for k, v in data.items():
        F = filename + '.' + k + '.npy'
        np.save(F, v)


def get_window_mask(data, window_size, CLS=True, SEP=-1):
    ''' For masking out the words in distance:
        only allow a word to attend to those near to it
        'near' means: within window_size words
    '''
    assert window_size >= 0, "Window size cannot be smaller than zero! "

    def Mask(seq):
        seq = [w.item() for w in seq]
        len_s = len(seq)
        mask = np.ones((len_s, len_s), dtype = np.uint8)
        if SEP >= 0:
            sep1 = seq.index(SEP)
            sep2 = len_s - 1
        for idx in range(len_s):
            if CLS and idx == 0:    # [CLS] can attend to all words in all sentences
                for i in range(0, len_s):
                    mask[idx][i] = 0
            elif SEP >= 0 and idx == sep1:  # [SEP] can attend to [CLS] and all words within the sentence
                for i in range(0, sep1 + 1):
                    mask[idx][i] = 0
            elif SEP >= 0 and idx == sep2:
                mask[idx][0] = 0
                for i in range(sep1 + 1, sep2 + 1):
                    mask[idx][i] = 0
            else:
                if CLS:
                    mask[idx][0] = 0   # all words can attend to [CLS]
                for i in range(idx - window_size, idx + window_size + 1):
                    if i >= 0 and i < len_s:
                        if SEP < 0 or ((idx <= sep1 and i <= sep1) or (idx > sep1 and i > sep1)):
                            mask[idx][i] = 0   # all words can attend to current window within the sentence
                if SEP >= 0 and idx <= sep1:    # all words can attend to [SEP] within the sentence
                    mask[idx][sep1] = 0
                elif SEP >= 0:
                    mask[idx][sep2] = 0
        return mask.reshape(len_s * len_s)
        
    masks = np.asarray([Mask(seq) for seq in tqdm(data)])
    return masks


def get_hack_window_mask(data, window_size, CLS=True, SEP=-1):
    ''' For masking out the neighbor words:
        only allow a word to attend to those in distance
        'in distance' means: most window-size farest words
    '''
    assert window_size >= 0, "Window size cannot be smaller than zero! "

    def Mask(seq):
        seq = [w.item() for w in seq]
        len_s = len(seq)
        distance = max((len_s - window_size * 2 - 4) // 2, window_size)
        mask = np.ones((len_s, len_s), dtype = np.uint8)
        if SEP >= 0:
            sep1 = seq.index(SEP)
            sep2 = len_s - 1
        for idx in range(len_s):
            if CLS and idx == 0:    # [CLS] can attend to all words in all sentences
                for i in range(0, len_s):
                    mask[idx][i] = 0
            elif SEP >= 0 and idx == sep1:  # [SEP] can attend to [CLS] and all words within the sentence
                for i in range(0, sep1 + 1):
                    mask[idx][i] = 0
            elif SEP >= 0 and idx == sep2:
                mask[idx][0] = 0
                for i in range(sep1 + 1, sep2 + 1):
                    mask[idx][i] = 0
            else:
                if CLS:
                    mask[idx][0] = 0   # all words can attend to [CLS]
                for i in range(idx - distance - window_size, idx + distance + window_size + 1):
                    # all words can attend to words in distance within the sentence
                    if abs(i - idx) >= distance and i >= 0 and i < len_s:
                        if SEP < 0 or ((idx <= sep1 and i <= sep1) or (idx > sep1 and i > sep1)):
                            mask[idx][i] = 0
                mask[idx][idx] = 0     # all words can attend to themselves
                if SEP >= 0 and idx <= sep1:    # all words can attend to [SEP] within the sentence
                    mask[idx][sep1] = 0
                elif SEP >= 0:
                    mask[idx][sep2] = 0
                    
        return mask.reshape(len_s * len_s)
        
    masks = np.asarray([Mask(seq) for seq in tqdm(data)])
    return masks


def get_random_mask(data, portion):
    ''' For randomly masking out words '''
    def Mask(seq):
        seq = [w.item() for w in seq]
        len_s = len(seq)
        mask = np.ones((len_s, len_s), dtype = np.uint8)

        candidates = [(k, v) for k in range(len_s) for v in range(0, k)]
        num = int(len(candidates) * portion)
        attend = random.sample(candidates, num)

        for k, v in attend:
            mask[k][v] = 0
            mask[v][k] = 0
        for i in range(len_s):
            mask[i][i] = 0
        
        return mask.reshape(len_s * len_s)
    
    masks = np.asarray([Mask(seq) for seq in tqdm(data)])
    return masks


def get_mask(data, opt):
    if opt.mode == 'window':
        mask = {k:get_window_mask(v, opt.window_size, CLS=opt.CLS, SEP=opt.SEP) for k,v in data.items()}
    elif opt.mode == 'hack-window':
        mask = {k:get_hack_window_mask(v, opt.window_size, CLS=opt.CLS, SEP=opt.SEP) for k,v in data.items()}
    elif opt.mode == 'random':
        mask = {k:get_random_mask(v, opt.portion) for k,v in data.items()}
    
    return mask


def main(opt):
    data = load_file(opt.input)
    mask = get_mask(data, opt) 
    dump_file(mask, opt.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mask_self_attn.py')
    parser.add_argument('-input', required=True)
    parser.add_argument('-output', required=True)
    parser.add_argument('-mode', default='window', choices=['window', 'hack-window', 'random'])

    parser.add_argument('-CLS', type=bool, default=1)
    parser.add_argument('-SEP', type=int, default=-1)
    parser.add_argument('-window_size', type=int, default=3)

    parser.add_argument('-portion', type=float, default=0.75)

    opt = parser.parse_args()

    main(opt)