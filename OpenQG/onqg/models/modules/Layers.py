''' Define the Layers '''
import torch
import torch.nn as nn
from onqg.models.modules.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from onqg.models.modules.Attention import GatedSelfAttention
import onqg.dataset.Constants as Constants


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, slf_attn, d_inner, n_head, d_k, d_v, 
                 dropout=0.1, attn_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = slf_attn
        if slf_attn == 'multi-head':
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        else:
            self.gated_slf_attn = GatedSelfAttention(d_model, d_k, dropout=attn_dropout)

    def forward(self, enc_input, src_seq, non_pad_mask=None, slf_attn_mask=None, layer_id=-1):
        if self.slf_attn == 'gated':
            mask = (src_seq == Constants.PAD).unsqueeze(2) if slf_attn_mask is None else slf_attn_mask
            enc_output, enc_slf_attn = self.gated_slf_attn(enc_input, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask

            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, 
                 addition_input=0, dropout=0.1, n_enc_layer=0,
                 layer_attn=False, two_step=False):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.layer_attn, self.two_step = layer_attn, two_step
        if layer_attn and self.two_step:
            self.enc_layer_attn = nn.ModuleList([
                MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) for _ in range(n_enc_layer)
            ])
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, addition_input=addition_input, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask
        
        if self.layer_attn and self.two_step:
            ### first step: respectively cross attention among each encoder-output with decoder-input ###
            enc_output = [enc_attn(dec_output, enc_output[i], enc_output[i], mask=dec_enc_attn_mask) 
                            for i, enc_attn in enumerate(self.enc_layer_attn)]
            enc_output = torch.stack([e_opt[0] for e_opt in enc_output], dim=2) # batch_size x tgt_dim x layer_num x dim

            ### second step: cross attention on layer-wise representation with decoder-input ###
            batch_size, layer_num, dim = enc_output.size(0), enc_output.size(2), enc_output.size(3)
            enc_output = enc_output.view(-1, layer_num, dim)    # (batch_size x tgt_len) x layer_num x dim
            dec_output = dec_output.view(-1, 1, dim)   # (batch_size x tgt_len) x 1 x dim
            dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output)
            dec_output = dec_output.view(batch_size, -1, dim)   # batch_size x tgt_len x dim
        else:
            dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
