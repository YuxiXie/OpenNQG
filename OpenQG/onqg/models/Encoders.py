import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import ConcatAttention, GatedSelfAttention
from onqg.models.modules.MaxOut import MaxOut
from onqg.models.modules.Layers import EncoderLayer

from onqg.utils.sinusoid import get_sinusoid_encoding_table
from onqg.utils.mask import get_attn_key_pad_mask, get_non_pad_mask, get_slf_window_mask

from pytorch_pretrained_bert import BertModel, GPT2Model


class RNNEncoder(nn.Module):
    """
    Input: (1) inputs['src_seq']
           (2) inputs['lengths'] 
           (3) inputs['feat_seqs']
    Output: (1) enc_output
            (2) hidden
    """
    def __init__(self, n_vocab, d_word_vec, d_model, n_layer,
                 brnn, rnn, feat_vocab, d_feat_vec, slf_attn, 
                 dropout):
        self.name = 'rnn'

        self.n_layer = n_layer
        self.num_directions = 2 if brnn else 1
        assert d_model % self.num_directions == 0, "d_model = hidden_size x direction_num"
        self.hidden_size = d_model // self.num_directions

        super(RNNEncoder, self).__init__()

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        input_size = d_word_vec

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            input_size += len(feat_vocab) * d_feat_vec
        
        self.slf_attn = slf_attn
        if slf_attn:
            self.gated_slf_attn = GatedSelfAttention(d_model)
        
        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=n_layer,
                               dropout=dropout, bidirectional=brnn, batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=n_layer,
                              dropout=dropout, bidirectional=brnn, batch_first=True)
        else:
            raise ValueError("Only support 'LSTM' and 'GRU' for RNN-based Encoder ")
    
    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['brnn'], opt['rnn'], opt['feat_vocab'], opt['d_feat_vec'], 
                   opt['slf_attn'], opt['dropout'])
    
    def forward(self, inputs):
        src_seq, lengths, feat_seqs = inputs['src_seq'], inputs['lengths'], inputs['feat_seqs']
        lengths = torch.LongTensor(lengths.data.view(-1).tolist())
        
        enc_input = self.word_emb(src_seq)
        if self.feature:
            feat_outputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_outputs = torch.cat(feat_outputs, dim=2)
            enc_input = torch.cat((enc_input, feat_outputs), dim=-1)
        
        enc_input = pack(enc_input, lengths, batch_first=True, enforce_sorted=False)
        enc_output, hidden = self.rnn(enc_input, None)
        enc_output = unpack(enc_output, batch_first=True)[0]

        if self.slf_attn:
            # mask = (src_seq == Constants.PAD).byte()
            mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
            enc_output, score = self.gated_slf_attn(enc_output, mask)
        
        return enc_output, hidden


class TransfEncoder(nn.Module):
    """
    Input: (1) inputs['src_seq']
           (2) inputs['src_pos']
           (3) inputs['feat_seqs']
    Output: (1) enc_output
            (2) hidden
    """
    def __init__(self, n_vocab, pretrained=None, model_name='default', d_word_vec=512, d_model=512, 
                 len_max_seq=512, n_layer=6, d_inner=2048, slf_attn='multi-head',
                 n_head=8, d_k=64, d_v=64, feat_vocab=None, d_feat_vec=32, 
                 layer_attn=False, slf_attn_mask='', dropout=0.1, attn_dropout=0.1):
        self.name = 'transf'
        self.model_type = model_name
        
        n_position = len_max_seq + 5

        super(TransfEncoder, self).__init__()

        self.feature = False if not feat_vocab else True
        self.layer_attn = layer_attn
        self.pretrained = pretrained
        self.defined_slf_attn_mask = True if slf_attn_mask else False

        if pretrained:
            return

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=Constants.PAD), 
            freeze=True
        )

        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, slf_attn, d_inner, n_head, d_k, d_v, dropout=dropout, attn_dropout=attn_dropout) for _ in range(n_layer)
        ])
    
    @classmethod
    def from_opt(cls, opt):
        if 'pretrained' not in opt:
            return cls(opt['n_vocab'], d_word_vec=opt['d_word_vec'], d_model=opt['d_model'], len_max_seq=opt['len_max_seq'],
                       n_layer=opt['n_layer'], d_inner=opt['d_inner'], n_head=opt['n_head'], slf_attn=opt['slf_attn'],
                       d_k=opt['d_k'], d_v=opt['d_v'], feat_vocab=opt['feat_vocab'], d_feat_vec=opt['d_feat_vec'], 
                       layer_attn=opt['layer_attn'], slf_attn_mask=opt['mask_slf_attn'],
                       dropout=opt['dropout'], attn_dropout=opt['attn_dropout'])
        elif opt['pretrained'].count('bert'):
            pretrained = BertModel.from_pretrained(opt['pretrained'])
            return cls(opt['n_vocab'], pretrained=pretrained, layer_attn=opt['layer_attn'], model_name='bert')
        elif opt['pretrained'].count('gpt2'):
            pretrained = GPT2Model.from_pretrained(opt['pretrained'])
            return cls(opt['n_vocab'], pretrained=pretrained, model_name='gpt2')
        else:
            raise ValueError("Other pretrained models haven't been supported yet")

    def forward(self, inputs, return_attns=False):
        if self.pretrained and self.model_type == 'bert':
            # src_seq, src_sep = inputs['src_seq'], inputs['src_sep']
            src_seq = inputs['src_seq']
        else:
            src_seq, src_pos, feat_seqs = inputs['src_seq'], inputs['src_pos'], inputs['feat_seqs']
        
        if self.pretrained:
            if self.model_type == 'bert':
                src_sep = None
                enc_outputs, *_ = self.pretrained(src_seq, token_type_ids=src_sep, output_all_encoded_layers=True)
                enc_output = enc_outputs[-1]
            elif self.model_type == 'gpt2':
                # enc_output, *_ = self.pretrained(src_seq, position_ids=src_pos)
                enc_output, *_ = self.pretrained(src_seq)
                hidden = [enc_output.transpose(0, 1)[-1] for _ in range(12)]
                return enc_output, hidden
        else:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
            non_pad_mask = get_non_pad_mask(src_seq)
            if self.defined_slf_attn_mask:
                def_slf_attn_mask = inputs['slf_attn_mask']
                slf_attn_mask = (slf_attn_mask + def_slf_attn_mask).gt(0)

            enc_output = self.word_emb(src_seq) + self.pos_emb(src_pos)
            if self.feature:
                assert feat_seqs is not None, "feature seq(s) must be given"
                feat_output = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_word_embs)]
                feat_output = torch.cat(feat_output, dim=2) 
                enc_output = torch.cat((enc_output, feat_output), dim=2)
            
            enc_outputs = []
            for layer_idx, enc_layer in enumerate(self.layer_stack):
                enc_output, _ = enc_layer(enc_output, src_seq, non_pad_mask=non_pad_mask, 
                                          slf_attn_mask=slf_attn_mask, layer_id=layer_idx)
                enc_outputs.append(enc_output)
        
        hidden = [layer_output.transpose(0, 1)[0] for layer_output in enc_outputs]
        if self.layer_attn:
            enc_output = enc_outputs
            
        return enc_output, hidden

