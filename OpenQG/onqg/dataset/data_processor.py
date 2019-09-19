import numpy as np

import torch

import onqg.dataset.Constants as Constants
from onqg.utils.mask import get_slf_attn_mask


def collate_fn(insts, sep=False, sep_id=Constants.SEP):
    """get src_seq, src_pos, src_sep (Tensor)"""
    pad_index = Constants.PAD if sep_id < 50000 else sep_id     # PAD tag in GPT2 is > 50000
    batch_seqs = insts    
    if sep:
        insts_tmp = [[w.item() for w in inst] for inst in insts]
        indexes = [inst.index(sep_id) for inst in insts_tmp]
        insts = [(inst[:indexes[i]+1], inst[indexes[i]+1:]) for i, inst in enumerate(insts)]
        insts = [[inst[0] for inst in insts], [inst[1] for inst in insts]]
    else:
        insts = [insts]
    batch_seq = [np.array([np.array(inst.cpu()) for inst in col]) for col in insts]
    
    if sep_id > 50000:     # PAD tag in GPT2 is > 50000
        batch_pos = [[[pos_i+1 if w_i != pad_index else 0 for pos_i,w_i in enumerate(inst + [10])] for inst in col] for col in batch_seq]
    else:
        batch_pos = [[[pos_i+1 if w_i != pad_index else 0 for pos_i,w_i in enumerate(inst)] for inst in col] for col in batch_seq]
    if sep:
        batch_pos = np.array([src + ans for src, ans in zip(batch_pos[0], batch_pos[1])])
    else:
        batch_pos = np.array(batch_pos[0])
    batch_pos = torch.LongTensor(batch_pos)
    
    if sep:
        batch_sep = [[[i for w in inst] for inst in col] for i, col in enumerate(batch_seq)]
        batch_sep = np.array([src + ans for src, ans in zip(batch_sep[0], batch_sep[1])])
        batch_sep = torch.LongTensor(batch_sep)
    
    del batch_seq
    batch_seq = batch_seqs

    if sep:
        return batch_seq, batch_pos, batch_sep
    else:
        return batch_seq, batch_pos

def preprocess_input(raw, device=None, return_sep=False, sep_id=Constants.SEP):
    """get src_seq, src_pos and (src_sep, max_length)"""
    raw = raw.transpose(0, 1)

    if return_sep:
        seq, pos, sep = collate_fn(raw, sep=return_sep, sep_id=sep_id)
        lengths = [[w.item() for w in sent if w.item() == 0] for sent in sep]     # TODO: fix this magic number
        lengths = [len(sent) for sent in lengths]
        max_length = max(lengths)
        if device:
            sep = sep.to(device)
        sep = (sep, max_length)
    else:
        seq, pos = collate_fn(raw, sep=return_sep)
    seq, pos = seq.to(device), pos.to(device)

    if return_sep:
        return seq, pos, sep
    else:
        return seq, pos

def get_sep_index(corpus):
    """get the index of the first [SEP] in each sentence"""
    if corpus is None:
        return None
    else:
        sep = [[w.item() for w in sent] for sent in corpus]
    indexes = [sent.index(1) for sent in sep]     # TODO: fix this magic number
    return indexes

def preprocess_batch(batch, separate=False, enc_rnn=False, dec_rnn=False,
                     feature=False, dec_feature=0, answer=False, ans_feature=False,
                     sep_id=Constants.SEP, copy=False, attn_mask=False, device=None):
    """Get a batch by indexing to the Dataset object, then preprocess it to get inputs for the model
    Input: batch
        raw-index: idxBatch
        src: (wrap(srcBatch), lengths)
        tgt: wrap(tgtBatch)
        copy: (wrap(copySwitchBatch), wrap(copyTgtBatch))
        feat: (tuple(wrap(x) for x in featBatches), lengths)
        ans: (wrap(ansBatch), ansLengths)
        ans_feat: (tuple(wrap(x) for x in ansFeatBatches), ansLengths)
    Output: 
        (1) inputs dict
            encoder: src_seq, lengths for rnn; 
                    src_seq, src_pos for transf; 
                    feat_seqs for both.
                    src_seq, src_sep for bert
            decoder: tgt_seq, src_indexes for rnn; 
                    tgt_seq, tgt_pos for transf; 
                    src_seq, feat_seqs for both.
            answer-encoder: src_seq, lengths, feat_seqs.
        (2) max_length: max_length of source text 
                        (except for answer part) in a batch
        (3) gold
        (4) (copy_gold, copy_switch)
    """
    inputs = {'encoder':{}, 'decoder':{}, 'answer-encoder':{}}

    src_seq, tgt_seq = batch['src'], batch['tgt']
    src_seq, lengths = src_seq[0], src_seq[1]        
    src_sep, max_length = None, 0
    if separate:
        #  for sentences contain [SEP] token
        src_seq, src_pos, src_sep = preprocess_input(src_seq, device=device, return_sep=separate, sep_id=sep_id)
        src_sep, max_length = src_sep[0], src_sep[1]
        inputs['encoder']['src_sep'] = src_sep
    else:
        src_seq, src_pos = preprocess_input(src_seq, device=device)
    tgt_seq, tgt_pos = preprocess_input(tgt_seq, device=device)
        
    if enc_rnn:
        inputs['encoder']['src_seq'], inputs['encoder']['lengths'] = src_seq, lengths
    else:
        inputs['encoder']['src_seq'], inputs['encoder']['src_pos'] = src_seq, src_pos
        if attn_mask:
            inputs['encoder']['slf_attn_mask'] = get_slf_attn_mask(attn_mask=batch['attn_mask'], lengths=lengths[0], 
                                                                   device=device)

    gold = tgt_seq[:, 1:]   # exclude [BOS] token
    if not dec_rnn:
        inputs['decoder']['tgt_seq'], inputs['decoder']['tgt_pos'] = tgt_seq[:, :-1], tgt_pos[:, :-1]   # exclude [EOS] token
    else:
        inputs['decoder']['tgt_seq'], inputs['decoder']['src_indexes'] = tgt_seq[:, :-1], get_sep_index(src_sep)
    inputs['decoder']['src_seq'] = inputs['encoder']['src_seq']

    src_feats, tgt_feats = None, None
    if feature:
        n_all_feature = len(batch['feat'][0])
        # split all features into src and tgt parts, src_feats are those embedded in the encoder
        src_feats = [feat.transpose(0, 1) for feat in batch['feat'][0][:n_all_feature - dec_feature]]
        if dec_feature:
            # dec_feature: the number of features embedded in the decoder
            tgt_feats = [feat.transpose(0, 1) for feat in batch['feat'][0][n_all_feature - dec_feature:]]
    inputs['encoder']['feat_seqs'], inputs['decoder']['feat_seqs'] = src_feats, tgt_feats

    ans_seq = None
    if answer:
        ans_seq = batch['ans']
        ans_seq, ans_lengths = ans_seq[0], ans_seq[1]
        ans_seq, _ = preprocess_input(ans_seq, device=device)
        ans_feats = None
        if ans_feature:
            ans_feats = [feat.transpose(0, 1) for feat in batch['ans_feat'][0]]
        inputs['answer-encoder']['feat_seqs'] = ans_feats
        inputs['answer-encoder']['src_seq'] = ans_seq
        inputs['answer-encoder']['lengths'] = ans_lengths
        
    copy_gold, copy_switch = None, None
    if copy:
        copy_gold, copy_switch = batch['copy'][1], batch['copy'][0]
        copy_gold, _ = preprocess_input(copy_gold, device=device)
        copy_switch, _ = preprocess_input(copy_switch, device=device)
        copy_gold, copy_switch = copy_gold[:, 1:], copy_switch[:, 1:]
        
    return inputs, max_length, gold, (copy_gold, copy_switch)