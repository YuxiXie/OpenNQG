import torch
import torch.nn as nn
from torch.autograd import Variable

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import ConcatAttention
from onqg.models.modules.MaxOut import MaxOut
from onqg.models.modules.DecAssist import StackedRNN, DecInit
from onqg.models.modules.Layers import DecoderLayer

from onqg.utils.mask import get_non_pad_mask, get_subsequent_mask, get_attn_key_pad_mask
from onqg.utils.sinusoid import get_sinusoid_encoding_table


class RNNDecoder(nn.Module):
    """
    Input: (1) inputs['tgt_seq']
           (2) inputs['src_seq']
           (3) inputs['src_indexes']
           (4) inputs['enc_output']
           (5) inputs['hidden']
           (6) inputs['feat_seqs']
    Output: (1) rst['pred']
            (2) rst['attn']
            (3) rst['context']
            (4) rst['copy_pred']; rst['copy_gate']
            (5) rst['coverage_pred']

    """
    def __init__(self, n_vocab, d_word_vec, d_model, n_layer,
                 rnn, d_k, feat_vocab, d_feat_vec, d_enc_model, n_enc_layer, 
                 input_feed, copy, separate, coverage, layer_attn,
                 maxout_pool_size, dropout, device=None):
        self.name = 'rnn'

        super(RNNDecoder, self).__init__()

        self.n_layer = n_layer
        self.layer_attn = layer_attn
        self.separate = separate
        self.coverage = coverage
        self.copy = copy
        self.maxout_pool_size = maxout_pool_size
        input_size = d_word_vec

        self.input_feed = input_feed
        if input_feed:
            input_size += d_enc_model

        self.decInit = DecInit(d_enc=d_enc_model, d_dec=d_model, n_enc_layer=n_enc_layer)

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            # input_size += len(feat_vocab) * d_feat_vec  # PS: only for test !!!
        feat_size = len(feat_vocab) * d_feat_vec if self.feature else 0

        self.d_enc_model = d_enc_model

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.rnn = StackedRNN(n_layer, input_size, d_model, dropout, rnn=rnn)
        self.attn = ConcatAttention(d_enc_model + feat_size, d_model, d_k, coverage)

        self.readout = nn.Linear((d_word_vec + d_model + self.d_enc_model), d_model)
        self.maxout = MaxOut(maxout_pool_size)

        if copy:
            self.copy_switch = nn.Linear(d_enc_model + d_model, 1)
        
        self.hidden_size = d_model
        self.dropout = nn.Dropout(dropout)
        self.device = device

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['rnn'], opt['d_k'], opt['feat_vocab'], opt['d_feat_vec'], 
                   opt['d_enc_model'], opt['n_enc_layer'], opt['input_feed'], opt['copy'], opt['separate'],
                   opt['coverage'], opt['layer_attn'], opt['maxout_pool_size'], opt['dropout'], 
                   opt['device'])

    def attn_init(self, context):
        if isinstance(context, list):
            context = context[-1]
        if isinstance(context, tuple):
            context = torch.cat(context, dim=-1)
        batch_size = context.size(0)
        hidden_sizes = (batch_size, self.d_enc_model)
        return Variable(context.data.new(*hidden_sizes).zero_(), requires_grad=False)

    def forward(self, inputs, max_length=300):
        tgt_seq, src_seq, src_indexes = inputs['tgt_seq'], inputs['src_seq'], inputs['src_indexes']
        enc_output, hidden, feat_seqs = inputs['enc_output'], inputs['hidden'], inputs['feat_seqs']

        src_pad_mask = Variable(src_seq.data.eq(50256).float(), requires_grad=False, volatile=False)    # TODO: fix this magic number later
        if self.layer_attn:
            n_enc_layer = len(enc_output)
            src_pad_mask = src_pad_mask.repeat(1, n_enc_layer)
            enc_output = torch.cat(enc_output, dim=1)
        
        feat_inputs = None
        if self.feature:
            feat_inputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, n_enc_layer, 1)
            # enc_output = torch.cat((enc_output, feat_inputs), dim=2)    # PS: only for test !!!

        dec_outputs, coverage_output, copy_output, copy_gate_output = [], [], [], []
        cur_context = self.attn_init(enc_output)
        hidden = self.decInit(hidden).unsqueeze(0)
        tmp_context, tmp_coverage = None, None

        dec_input = self.word_emb(tgt_seq)
        
        self.attn.apply_mask(src_pad_mask)

        dec_input = dec_input.transpose(0, 1)
        for seq_idx, dec_input_emb in enumerate(dec_input.split(1)):
            dec_input_emb = dec_input_emb.squeeze(0)
            raw_dec_input_emb = dec_input_emb
            if self.input_feed:
                dec_input_emb = torch.cat((dec_input_emb, cur_context), dim=1)
            dec_output, hidden = self.rnn(dec_input_emb, hidden)

            if self.coverage:
                if tmp_coverage is None:
                    tmp_coverage = Variable(torch.zeros((enc_output.size(0), enc_output.size(1))))
                    if self.device:
                        tmp_coverage = tmp_coverage.to(self.device)
                cur_context, attn, tmp_context, next_coverage = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                                          coverage=tmp_coverage, feat_inputs=feat_inputs,
                                                                          feature=self.feature)
                avg_tmp_coverage = tmp_coverage / max(1, seq_idx)
                coverage_loss = torch.sum(torch.min(attn, avg_tmp_coverage), dim=1)
                tmp_coverage = next_coverage
                coverage_output.append(coverage_loss)
            else:
                cur_context, attn, tmp_context = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                           feat_inputs=feat_inputs, feature=self.feature)
            
            if self.copy:
                copy_prob = self.copy_switch(torch.cat((dec_output, cur_context), dim=1))
                copy_prob = torch.sigmoid(copy_prob)

                if self.layer_attn:
                    attn = attn.view(attn.size(0), n_enc_layer, -1)
                    attn = attn.sum(1)
                
                if self.separate:
                    out = torch.zeros([len(attn), max_length], device=self.device if self.device else None)
                    for i in range(len(attn)):
                        data_length = src_indexes[i]
                        out[i].narrow(0, 1, data_length - 1).copy_(attn[i][1:src_indexes[i]])
                    attn = out
                    norm_term = attn.sum(1, keepdim=True)
                    attn = attn / norm_term
                
                copy_output.append(attn)
                copy_gate_output.append(copy_prob)
            
            readout = self.readout(torch.cat((raw_dec_input_emb, dec_output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            
            dec_outputs.append(output)
        
        dec_output = torch.stack(dec_outputs).transpose(0, 1)

        rst = {}
        rst['pred'], rst['attn'], rst['context'] = dec_output, attn, cur_context
        if self.copy:
            copy_output = torch.stack(copy_output).transpose(0, 1)
            copy_gate_output = torch.stack(copy_gate_output).transpose(0, 1)
            rst['copy_pred'], rst['copy_gate'] = copy_output, copy_gate_output
        if self.coverage:
            coverage_output = torch.stack(coverage_output).transpose(0, 1)
            rst['coverage_pred'] = coverage_output
        return rst


class TransfDecoder(nn.Module):
    def __init__(self, n_vocab, len_max_seq, d_word_vec, d_model, n_layer,
                 d_inner, n_head, d_k, d_v, layer_attn, n_enc_layer, 
                 feat_vocab, d_feat_vec, maxout_pool_size, dropout):
        self.name = 'transf'

        super(TransfDecoder, self).__init__()

        n_position = len_max_seq + 5
        self.layer_attn = layer_attn

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=Constants.PAD),
            freeze=True
        )

        self.feature = False if not feat_vocab else True
        self.d_feat = len(feat_vocab) * d_feat_vec if self.feature else 0
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, addition_input=self.d_feat,
                         dropout=dropout, layer_attn=layer_attn, n_enc_layer=n_enc_layer)
            for _ in range(n_layer)
        ])

        self.maxout = MaxOut(maxout_pool_size)
        self.maxout_pool_size = maxout_pool_size
    
    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['len_max_seq'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['d_inner'], opt['n_head'], opt['d_k'], opt['d_v'], opt['layer_attn'], opt['n_enc_layer'],
                   opt['feat_vocab'], opt['d_feat_vec'], opt['maxout_pool_size'], opt['dropout'])
    
    def forward(self, inputs, max_length=300, return_attns=False):
        tgt_seq, tgt_pos, feat_seqs = inputs['tgt_seq'], inputs['tgt_pos'], inputs['feat_seqs']
        src_seq, enc_output, _ = inputs['src_seq'], inputs['enc_output'], inputs['hidden']

        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        if self.layer_attn:
            enc_output = torch.stack(enc_output)    # layer_num x batch_size x src_len x dim
            batch_size, layer_num, dim = enc_output.size(1), enc_output.size(0), enc_output.size(-1)

            layer_src_seq = src_seq.unsqueeze(1).repeat(1, layer_num, 1) # batch_size x layer_num x src_len
            layer_src_seq = layer_src_seq.contiguous().view(batch_size, -1)  # batch_size x (layer_num x src_len)
            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=layer_src_seq, seq_q=tgt_seq)
            enc_output = enc_output.permute(1, 0, 2, 3).contiguous().view(batch_size, -1, dim)  # batch_size x (layer_num x src_len) x dim
        
        if self.feature:
            feat_inputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, layer_num, 1)
            enc_output = torch.cat((enc_output, feat_inputs), dim=2)
        
        dec_output = self.word_emb(tgt_seq) + self.pos_emb(tgt_pos)
        for dec_layer in self.layer_stack:
            dec_output, *_ = dec_layer(dec_output, enc_output, 
                                       non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                       dec_enc_attn_mask=dec_enc_attn_mask)
        
        dec_output = self.maxout(dec_output)
        
        rst = {'pred':dec_output}
        return rst
