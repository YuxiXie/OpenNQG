import math
import torch
import argparse

import pargs
import onqg.dataset.Constants as Constants
from onqg.dataset.Vocab import Vocab


def load_vocab(filename):
    vocab_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    text = [word.split(' ') for word in text]
    vocab_dict = {word[0]:word[1:] for word in text}
    vocab_dict = {k:[float(d) for d in v] for k,v in vocab_dict.items()}
    return vocab_dict

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    data = data.split('\n')
    data = [sent.strip().split() for sent in data]
    return data

def filter_data(data_list, opt):
    data_num = len(data_list)
    idx = list(range(len(data_list[0])))
    rst = [[] for _ in range(data_num)]
    
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        # add [CLS] at the beginning for TransfEncoder - assume seperate answer given for RNNEncoder
        src_len = len(src) if opt.answer == 'enc' else len(src) + 1
        if opt.answer == 'sep':
            ans = data_list[2][i]
            src_len += len(ans) + 2   # add [SEP] for TransfEncoder if answer is given
        if src_len <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length:
            if len(src) * len(tgt) > 0:
                if opt.answer == 'enc':
                    rst[0].append(src)
                else:
                    rst[0].append([Constants.CLS_WORD] + src)
                rst[1].append([Constants.BOS_WORD] + tgt + [Constants.EOS_WORD])
                for j in range(2, data_num):
                    sent = data_list[j][i]
                    rst[j].append(sent)
            else:
                with open('dump_idx.txt', 'a', encoding='utf-8') as f:
                    f.write(str(i) + '\t')
        else:
            with open('dump_idx.txt', 'a', encoding='utf-8') as f:
                f.write(str(i) + '\t')
    
    with open('dump_idx.txt', 'a', encoding='utf-8') as f:
        f.write('\n')

    numBatches = math.floor(len(rst[0]) / opt.batch_size)
    length = numBatches * opt.batch_size
    rst = [d[:length] for d in rst]
    print("change data size from " + str(len(idx)) + " to " + str(len(rst[0])))
    return rst

def convert_word_to_idx(text, vocab, bert=False, sep=False):

    def lower_sent(sent):
        for idx, w in enumerate(sent):
            if w not in ['[BOS]', '[EOS]', '[PAD]', '[SEP]', '[UNK]', '[CLS]']:
                sent[idx] = w.lower()
        return sent
    
    tokens = [lower_sent(sent) for sent in text]

    if bert:
        text = [' '.join(sent) for sent in tokens]
        if sep:
            text = [sent.strip().split(' ' + Constants.SEP_WORD + ' ') for sent in text]
            tokens = [vocab.tokenizer.tokenize(sent[0].strip()) + [Constants.SEP_WORD] + vocab.tokenizer.tokenize(sent[1].strip()) for sent in text]
        else:
            tokens = [vocab.tokenizer.tokenize(sent) for sent in text]
    
    indexes = [vocab.convertToIdx(sent) for sent in tokens]
    return indexes, tokens

def get_embedding(vocab_dict, vocab):

    def get_vector(idx):
        word = vocab.idxToLabel[idx]
        if idx in vocab.special or word not in vocab_dict:
            vector = torch.tensor([])
            vector = vector.new_full((opt.word_vec_size,), 1.0)
            vector.normal_(0, math.sqrt(6 / (1 + vector.size(0))))
        else:
            vector = torch.Tensor(vocab_dict[word])
        return vector
    
    embedding = [get_vector(idx) for idx in range(vocab.size)]
    embedding = torch.stack(embedding)
    
    print(embedding.size())

    return embedding

def get_data(files, opt):
    src, tgt = load_file(files['src']), load_file(files['tgt'])
    data_list = [src, tgt]
    if opt.answer:
        data_list.append(load_file(files['ans']))
    if opt.feature:
        data_list += [load_file(filename) for filename in files['feats']]
    if opt.ans_feature:
        data_list += [load_file(filename) for filename in files['ans_feats']]
    
    data_list = filter_data(data_list, opt)

    rst = {'src':data_list[0], 'tgt':data_list[1]}
    i = 2
    if opt.answer:
        rst['ans'] = data_list[i]
        i += 1
    if opt.feature:
        rst['feats'] = [data_list[i] for i in range(i, i + len(files['feats']))]
        i += 1
    if opt.ans_feature:
        rst['ans_feats'] = [data_list[i] for i in range(i, i + len(files['ans_feats']))]        
    return rst

def merge_ans(src, ans):
    rst = [s + [Constants.SEP_WORD] + a + [Constants.SEP_WORD] for s, a in zip(src, ans)]
    return rst

def wrap_copy_idx(splited, tgt, tgt_vocab, bert):
    
    def map_src(sp):
        sp_split = {}
        if bert:
            tmp_idx = 0
            tmp_word = ''
            sep_idx = sp.index('[SEP]')
            for i, w in enumerate(sp):
                if i >= sep_idx:
                    break
                if not w.startswith('##'):
                    if tmp_word:
                        sp_split[tmp_word] = tmp_idx
                    tmp_word = w
                    tmp_idx = i
                else:
                    tmp_word += w.lstrip('##')
            sp_split[tmp_word] = tmp_idx
        else:
            if '[SEP]' not in sp:
                import ipdb; ipdb.set_trace()
            sep_idx = sp.index('[SEP]')
            sp_split = {w:idx for idx, w in enumerate(sp) if idx < sep_idx}
        return sp_split

    def wrap_sent(sp, t):
        sp_dict = map_src(sp)
        swt, cpt = [0 for w in t], [0 for w in t]
        for i, w in enumerate(t):
            if w not in tgt_vocab.labelToIdx or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
                if w in sp_dict:
                    swt[i] = 1
                    cpt[i] = sp_dict[w]
        return torch.Tensor(swt), torch.LongTensor(cpt)
    
    copy = [wrap_sent(sp, t) for sp, t in zip(splited, tgt)]
    switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
    return [switch, cp_tgt]
    
def main(opt):
    #========== get data ==========#
    train_files = {'src':opt.train_src, 'tgt':opt.train_tgt}
    valid_files = {'src':opt.valid_src, 'tgt':opt.valid_tgt}
    if opt.feature:
        assert len(opt.train_feats) == len(opt.valid_feats) and len(opt.train_feats) > 0
        train_files['feats'], valid_files['feats'] = opt.train_feats, opt.valid_feats
    if opt.answer:
        assert opt.train_ans and opt.valid_ans, "Answer files of train and valid must be given"
        train_files['ans'], valid_files['ans'] = opt.train_ans, opt.valid_ans
    if opt.ans_feature:
        assert len(opt.train_ans_feats) == len(opt.valid_ans_feats) and len(opt.train_ans_feats) > 0 and opt.answer == 'enc'
        train_files['ans_feats'], valid_files['ans_feats'] = opt.train_ans_feats, opt.valid_ans_feats
    
    train_data = get_data(train_files, opt)
    valid_data = get_data(valid_files, opt)

    train_src, train_tgt = train_data['src'], train_data['tgt']
    valid_src, valid_tgt = valid_data['src'], valid_data['tgt']
    train_feats = train_data['feats'] if opt.feature else None
    valid_feats = valid_data['feats'] if opt.feature else None
    train_ans = train_data['ans'] if opt.answer else None
    valid_ans = valid_data['ans'] if opt.answer else None
    train_ans_feats = train_data['ans_feats'] if opt.ans_feature else None
    valid_ans_feats = valid_data['ans_feats'] if opt.ans_feature else None

    #========== build vocabulary ==========#
    sep = True if opt.answer == 'sep' else False
    if opt.answer == 'sep':
        train_src = merge_ans(train_src, train_ans)
        valid_src = merge_ans(valid_src, valid_ans)

    pre_trained_vocab = load_vocab(opt.pre_trained_vocab) if opt.pre_trained_vocab else None  
    if opt.share_vocab:
        assert not opt.bert
        print("build src & tgt vocabulary")
        corpus = train_src + train_tgt
        corpus = corpus + train_ans if opt.answer == 'enc' else corpus
        options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'transf':opt.answer != 'enc', 'separate':sep, 'tgt':True,
                   'size':max(opt.src_vocab_size, opt.tgt_vocab_size),
                   'frequency':min(opt.src_words_min_frequency, opt.tgt_words_min_frequency)}
        vocab = Vocab.from_opt(corpus=corpus, opt=options)
        src_vocab = tgt_vocab = vocab
        ans_vocab = vocab if opt.answer == 'enc' else None
    else:
        print("build src vocabulary")
        if opt.bert:
            options = {'transf':opt.answer != 'enc', 'separate':sep, 'tgt':False}
            src_vocab = Vocab.from_opt(pretrained=opt.bert_vocab, opt=options)
        else:
            corpus = train_src + train_ans if opt.answer == 'enc' else train_src
            options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'transf':opt.answer != 'enc', 'separate':sep, 'tgt':False, 
                       'size':opt.src_vocab_size, 'frequency':opt.src_words_min_frequency}
            src_vocab = Vocab.from_opt(corpus=corpus, opt=options)
            ans_vocab = src_vocab if opt.answer == 'enc' else None
        print("build tgt vocabulary")
        options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'transf':False, 'separate':False, 'tgt':True, 
                   'size':opt.tgt_vocab_size, 'frequency':opt.tgt_words_min_frequency}
        tgt_vocab = Vocab.from_opt(corpus=train_tgt, opt=options)
    
    options = {'lower':False, 'mode':'size', 'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency,
               'transf':opt.answer != 'enc', 'separate':sep, 'tgt':False}
    feats_vocab = [Vocab.from_opt(corpus=feat, opt=options) for feat in train_feats] if opt.feature else None
    ans_feats_vocab = [Vocab.from_opt(corpus=feat, opt=options) for feat in train_ans_feats] if opt.ans_feature else None
        
    #========== word to index ==========#
    train_src_idx, train_src_tokens = convert_word_to_idx(train_src, src_vocab, bert=opt.bert, sep=sep)
    train_tgt_idx, train_tgt_tokens = convert_word_to_idx(train_tgt, tgt_vocab)
    valid_src_idx, valid_src_tokens = convert_word_to_idx(valid_src, src_vocab, bert=opt.bert, sep=sep)
    valid_tgt_idx, valid_tgt_tokens = convert_word_to_idx(valid_tgt, tgt_vocab)

    train_copy = wrap_copy_idx(train_src_tokens, train_tgt_tokens, tgt_vocab, opt.bert) if opt.copy else [None, None]
    valid_copy = wrap_copy_idx(valid_src_tokens, valid_tgt_tokens, tgt_vocab, opt.bert) if opt.copy else [None, None]
    train_copy_switch, train_copy_tgt = train_copy[0], train_copy[1]
    valid_copy_switch, valid_copy_tgt = valid_copy[0], valid_copy[1]

    train_ans_idx = convert_word_to_idx(train_ans, ans_vocab)[0] if opt.answer == 'enc' else None
    valid_ans_idx = convert_word_to_idx(valid_ans, ans_vocab)[0] if opt.answer == 'enc' else None
    train_feat_idxs = [convert_word_to_idx(feat, vocab)[0] for feat, vocab 
                        in zip(train_feats, feats_vocab)] if opt.feature else None
    valid_feat_idxs = [convert_word_to_idx(feat, vocab)[0] for feat, vocab 
                        in zip(valid_feats, feats_vocab)] if opt.feature else None
    train_ans_feat_idxs = [convert_word_to_idx(feat, vocab)[0] for feat, vocab 
                            in zip(train_ans_feats, ans_feats_vocab)] if opt.ans_feature else None 
    valid_ans_feat_idxs = [convert_word_to_idx(feat, vocab)[0] for feat, vocab 
                            in zip(valid_ans_feats, ans_feats_vocab)] if opt.ans_feature else None

    #========== prepare pretrained vetors ==========#
    if pre_trained_vocab:
        pre_trained_src_vocab = None if opt.bert else get_embedding(pre_trained_vocab, src_vocab)
        pre_trained_tgt_vocab = get_embedding(pre_trained_vocab, tgt_vocab)
        pre_trained_ans_vocab = get_embedding(pre_trained_vocab, ans_vocab) if opt.answer == 'enc' else None
        pre_trained_vocab = {'src':pre_trained_src_vocab, 'tgt':pre_trained_tgt_vocab, 'ans':pre_trained_ans_vocab}

    #========== save data ===========#
    data = {'settings': opt, 
            'dict': {'src': src_vocab,
                     'tgt': tgt_vocab,
                     'ans': ans_vocab if opt.answer == 'enc' else None,
                     'feature': feats_vocab, 
                     'ans_feature': ans_feats_vocab,
                     'pre-trained': pre_trained_vocab
            },
            'train': {'src': train_src_idx,
                      'tgt': train_tgt_idx,
                      'ans': train_ans_idx,
                      'feature': train_feat_idxs,
                      'ans_feature': train_ans_feat_idxs, 
                      'copy':{'switch':train_copy_switch,
                              'tgt':train_copy_tgt}
            },
            'valid': {'src': valid_src_idx,
                      'tgt': valid_tgt_idx,
                      'ans': valid_ans_idx,
                      'feature': valid_feat_idxs,
                      'ans_feature': valid_ans_feat_idxs, 
                      'copy':{'switch':valid_copy_switch,
                              'tgt':valid_copy_tgt},
                      'tokens':{'src': valid_src_tokens,
                                'tgt': valid_tgt_tokens}
            }
        }
    
    torch.save(data, opt.save_data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess.py')
    pargs.add_options(parser)
    opt = parser.parse_args()
    main(opt)
