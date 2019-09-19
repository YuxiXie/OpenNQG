import os
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import cuda

from pytorch_pretrained_bert import BertTokenizer, BertModel

from onqg.dataset.Dataset import Dataset
from onqg.dataset.data_processor import preprocess_input
from onqg.dataset.Vocab import Vocab
from onqg.utils.train.Optim import Optimizer
from onqg.utils.train.Loss import NLLLoss



def filter_data(fsrc, ftgt, tokenizer):

    def load_file(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')
        return data
    
    def get_tag(tokens, raw):
        word, idraw = '', -1
        rst = []
        for w in tokens:
            idx = 3 if w.startswith('##') else 2
            if idx == 3:
                word = word + w.lstrip('##')
            elif idx == 2:
                if word == '' or len(word) == len(raw[idraw]) or word == '[UNK]':
                    idraw += 1
                    word = w
                else:
                    idx = 3
                    word = word + w
            rst.append(idx)
        rst = torch.tensor(rst, dtype=torch.uint8)
        return rst
    
    src, tgt = load_file(fsrc), load_file(ftgt)
    Ss, Ts, Idx = [], [], []
    for s, t in zip(src, tgt):
        s = s.strip().split()
        t = t.strip().split()
        if len(s) == len(t) and len(s) > 0 and len(s) <= 128:
            raw = s
            s = tokenizer.tokenize(' '.join(s))
            try:
                idx = get_tag(s, raw)
                b = [w for w in idx if w == 2]
                if len(b) == len(t):
                    Ss.append(s)
                    Ts.append(t)
                    Idx.append(idx)
            except:
                print(' '.join(raw))
            
    print(len(Ss))         
    return {'src':Ss, 'tgt':Ts, 'idx':Idx}


class NERTagger(nn.Module):
    def __init__(self, encoder, classifier, device):
        super(NERTagger, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
    
    def forward(self, src_seq, src_index):
        src_seq, src_index = src_seq.transpose(0, 1), src_index.transpose(0, 1)
        enc_outputs, *_ = self.encoder(src_seq, output_all_encoded_layers=True)
        enc_outputs = enc_outputs[-1]

        hidden_states = []
        for indexes, enc_batch in zip(src_index, enc_outputs):
            tmp_sent = []

            tmp_output = None
            for idx, enc_output in zip(indexes, enc_batch):
                if idx.item() == 2:
                    if tmp_output is not None:
                        tmp_sent.append(tmp_output)
                        tmp_output = None
                    tmp_output = enc_output
                elif idx.item() == 3:
                    tmp_output = tmp_output + enc_output
            if tmp_output is not None:
                tmp_sent.append(tmp_output)

            tmp_sent = torch.stack(tmp_sent, dim=0)
            hidden_states.append(tmp_sent)
        hidden_states = torch.cat(hidden_states, dim=0).to(self.device)

        output = self.classifier(hidden_states)

        return output


def main(opt):
    tokenizer = BertTokenizer.from_pretrained(opt.pre_model)
    ###========== Load Data ==========###
    train_data = filter_data(opt.train_src, opt.train_tgt, tokenizer)
    valid_data = filter_data(opt.valid_src, opt.valid_tgt, tokenizer)
    ###========== Get Index ==========###
    options = {'transf':False, 'separate':False, 'tgt':False}
    src_vocab = Vocab.from_opt(pretrained=opt.pre_model, opt=options)
    options = {'lower':False, 'mode':'size', 'size':1000, 'frequency':1,
               'transf':False, 'separate':False, 'tgt':False}
    tgt_vocab = Vocab.from_opt(corpus=train_data['tgt'], opt=options)
    train_src_idx = [src_vocab.convertToIdx(sent) for sent in train_data['src']]
    valid_src_idx = [src_vocab.convertToIdx(sent) for sent in valid_data['src']]
    train_tgt_idx = [tgt_vocab.convertToIdx(sent) for sent in train_data['tgt']]
    valid_tgt_idx = [tgt_vocab.convertToIdx(sent) for sent in valid_data['tgt']]
    ###========== Get Data ==========###
    train_data = Dataset({'src':train_src_idx, 'tgt':train_tgt_idx, 'feature':[train_data['idx']]}, 
                         opt.batch_size, feature=True, opt_cuda=opt.gpus)
    valid_data = Dataset({'src':valid_src_idx, 'tgt':valid_tgt_idx, 'feature':[valid_data['idx']]}, 
                         opt.batch_size, feature=True, opt_cuda=opt.gpus)
    opt.tgt_vocab_size = tgt_vocab.size
    ###========== Prepare Model ==========###
    device = torch.device('cuda')
    encoder = BertModel.from_pretrained(opt.pre_model)
    classifier = nn.Sequential(
        nn.Linear(768 // opt.maxout_pool_size, opt.tgt_vocab_size),     # TODO: fix this magic number later (hidden size of the model)
        nn.Softmax(dim=1)
    )
    model = NERTagger(encoder, classifier, device).to(device)
    for _, para in model.classifier.named_parameters():
        if para.dim() == 1:
            para.data.normal_(0, math.sqrt(6 / (1 + para.size(0))))
        else:
            nn.init.xavier_normal(para, math.sqrt(3))
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus)
    ###========== Prepare for training ==========###
    opt.optim = 'adam'
    opt.decay_method = ''
    opt.learning_rate = 3e-5
    opt.learning_rate_decay = 1
    opt.decay_steps = 10000000
    opt.start_decay_steps = 10000000000
    opt.max_grad_norm = 5
    opt.max_weight_value = 20
    opt.decay_bad_cnt = 5
    optimizer = Optimizer.from_opt(model, opt)
    
    weight = torch.ones(opt.tgt_vocab_size)
    weight[0] = 0       # TODO: fix this magic number later (PAD)
    loss = NLLLoss(opt, weight, size_average=False)
    if opt.gpus:
        loss.cuda()
    ###========== Training ==========###
    best_val = 0

    def eval_model(M, D, L):
        M.eval()

        all_loss, all_accu, all_words = 0, 0, 0
        for i in tqdm(range(len(D)), mininterval=2, desc='  - (Validation)  ', leave=False):
            B = D[i]
            s, t, sid = B['src'][0], B['tgt'], B['feat'][0][0]
            t = t.transpose(0, 1)
            P = M(s, sid)
            lv, G = L.cal_loss_ner(P, t)

            all_loss += lv.item()
            all_words += P.size(0)
            P = P.max(1)[1]
            n_correct = P.eq(G.view(-1))
            n_correct = n_correct.sum().item()
            all_accu += n_correct

        return all_loss/all_words, all_accu/all_words

    def save_model(M, score, best_val, opt):
        if score > best_val:
            model_to_save = M.module.encoder if hasattr(M, 'module') else M.encoder  # Only save the model it-self
            output_model_file = os.path.join(opt.output_dir, "pytorch_model_" + str(round(score * 100, 2)) + ".bin")
            torch.save(model_to_save.state_dict(), output_model_file)
        print('validation', score)

    for _ in range(opt.num_train_epochs):
        train_data.shuffle()
        model.train()
        batch_order = torch.randperm(len(train_data))
        loss_print, words_cnt, accuracy = 0, 0, 0
        for idx in tqdm(range(len(train_data)), mininterval=2, desc='  - (Training)  ', leave=False):
            batch_idx = batch_order[idx]
            batch = train_data[batch_idx]

            src, tgt, src_idx = batch['src'][0], batch['tgt'], batch['feat'][0][0]
            tgt = tgt.transpose(0, 1)

            out = model(src, src_idx)
            loss_val, gold = loss.cal_loss_ner(out, tgt)
            if len(opt.gpus) > 1:
                loss_val = loss_val.mean()  # mean() to average on multi-gpu.
            if math.isnan(loss_val.item()) or loss_val.item() > 1e20:
                print('catch NaN')
                import ipdb; ipdb.set_trace()
            loss_val.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_print += loss_val.item()
            words_cnt += out.size(0)
            pred = out.max(1)[1]
            n_correct = pred.eq(gold.view(-1))
            n_correct = n_correct.sum().item()
            accuracy += n_correct
            if idx % 1000 == 0:
                loss_print /= words_cnt
                accuracy /= words_cnt
                print('loss', loss_print)
                print('accuracy', accuracy)
                loss_val, words_cnt, accuracy = 0, 0, 0
                if idx % 2000 == 0:
                    loss_val, accuracy_val = eval_model(model, valid_data, loss)
                    save_model(model, accuracy_val, best_val, opt)
                    if accuracy_val > best_val:
                        best_val = accuracy_val
    
    model_to_save = model.module.encoder if hasattr(model, 'module') else model.encoder  # Only save the model it-self
    output_model_file = os.path.join(opt.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finetune_ner.py')

    parser.add_argument('-pre_model', type=str, choices=['bert-base-uncased'])
    parser.add_argument('-maxout_pool_size', type=int, default=1)

    parser.add_argument('-train_src', required=True, help='Path to training source')
    parser.add_argument('-train_tgt', required=True, help='Path to training target')
    parser.add_argument('-valid_src', required=True, help='Path to validation source')
    parser.add_argument('-valid_tgt', required=True, help='Path to validation target')
    parser.add_argument('-output_dir', required=True)

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-gpus', default=[], nargs='+', type=int)

    parser.add_argument('-gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('-num_train_epochs', type=int, default=5)
    parser.add_argument('-learning_rate', type=float, default=3e-5)
    parser.add_argument('-warmup_proportion', type=float, default=0.1)

    opt = parser.parse_args()
    if opt.gpus:
        cuda.set_device(opt.gpus[0])
    
    main(opt)
