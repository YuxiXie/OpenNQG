import math
import torch.nn as nn

from onqg.models.Models import OpenNQG
from onqg.models.Encoders import RNNEncoder, TransfEncoder
from onqg.models.Decoders import RNNDecoder, TransfDecoder


def build_encoder(opt, answer=False, separate=-1):
    feat_vocab = opt.feat_vocab
    if answer:
        feat_vocab = opt.ans_feat_vocab
    elif feat_vocab:
        n_all_feat = len(feat_vocab)
        feat_vocab = feat_vocab[:n_all_feat - opt.dec_feature]

    if opt.enc_rnn:
        options = {'n_vocab':opt.src_vocab_size, 'd_word_vec':opt.d_word_vec, 'd_model':opt.d_enc_model,
                   'n_layer':opt.n_enc_layer, 'brnn':opt.brnn, 'rnn':opt.enc_rnn, 'slf_attn':opt.slf_attn, 
                   'feat_vocab':feat_vocab, 'd_feat_vec':opt.d_feat_vec, 'dropout':opt.dropout}
        if answer:
            options['slf_attn'] = False

        model = RNNEncoder.from_opt(options)
    else:
        if opt.pretrained:
            options = {'pretrained':opt.pretrained, 'n_vocab':opt.src_vocab_size, 'layer_attn':opt.layer_attn}
        else:
            options = {'n_vocab':opt.src_vocab_size, 'd_word_vec':opt.d_word_vec, 'd_model':opt.d_enc_model, 
                       'len_max_seq':opt.max_token_src_len, 'n_layer':opt.n_enc_layer, 'd_inner':opt.d_inner, 
                       'slf_attn':opt.slf_attn_type, 'n_head':opt.n_head, 'd_k':opt.d_k, 'd_v':opt.d_v, 'feat_vocab':feat_vocab, 
                       'd_feat_vec':opt.d_feat_vec, 'layer_attn':opt.layer_attn, 'mask_slf_attn':opt.defined_slf_attn_mask,
                       'separate':separate, 'dropout':opt.dropout, 'attn_dropout':opt.attn_dropout}
        
        model = TransfEncoder.from_opt(options)

        if opt.pretrained:
            for para in model.parameters():
                para.requires_grad = False

    return model       


def build_decoder(opt, device):
    if opt.dec_feature:
        n_all_feat = len(opt.feat_vocab)
        feat_vocab = opt.feat_vocab[n_all_feat - opt.dec_feature:]
    else:
        feat_vocab = None
    
    d_enc_model = opt.d_enc_model if not opt.pretrained else 768        # TODO: fix this magic number later
    n_enc_layer = opt.n_enc_layer if not opt.pretrained else 12         # TODO: fix this magic number later
    
    if opt.dec_rnn:
        options = {'n_vocab':opt.tgt_vocab_size, 'd_word_vec':opt.d_word_vec, 'd_model':opt.d_dec_model,
                   'n_layer':opt.n_dec_layer, 'rnn':opt.dec_rnn, 'd_k':opt.d_k, 'feat_vocab':feat_vocab,
                   'd_feat_vec':opt.d_feat_vec, 'd_enc_model':d_enc_model, 'n_enc_layer':n_enc_layer,
                   'input_feed':opt.input_feed, 'copy':opt.copy, 'coverage':opt.coverage, 
                   'separate':opt.answer == 'sep', 'layer_attn':opt.layer_attn,
                   'maxout_pool_size':opt.maxout_pool_size, 'dropout':opt.dropout, 'device':device}
        model = RNNDecoder.from_opt(options)
    else:
        options = {'n_vocab':opt.tgt_vocab_size, 'len_max_seq':opt.max_token_tgt_len, 'd_word_vec':opt.d_word_vec, 
                   'd_model':opt.d_dec_model, 'n_layer':opt.n_dec_layer, 'd_inner':opt.d_inner, 'n_head':opt.n_head,
                   'd_k':opt.d_k, 'd_v':opt.d_v, 'layer_attn':opt.layer_attn, 'n_enc_layer':n_enc_layer, 
                   'feat_vocab':feat_vocab, 'd_feat_vec':opt.d_feat_vec, 'maxout_pool_size':opt.maxout_pool_size,
                   'dropout':opt.dropout}
        model = TransfDecoder.from_opt(options)
    
    return model


def initialize(model, opt):
    parameters_cnt = 0
    for name, para in model.named_parameters():
        if not opt.pretrained or name.count('encoder') == 0:
            if para.dim() == 1:
                para.data.normal_(0, math.sqrt(6 / (1 + para.size(0))))
            else:
                nn.init.xavier_normal(para, math.sqrt(3))
            size = list(para.size())
            local_cnt = 1
            for d in size:
                local_cnt *= d
            parameters_cnt += local_cnt
    
    if opt.pre_trained_vocab:
        assert opt.d_word_vec == 300, "Dimension of word vectors must equal to that of pretrained word-embedding"
        if not opt.pretrained:
            model.encoder.word_emb.weight.data.copy_(opt.pre_trained_src_emb)
        if opt.answer == 'enc':
            model.answer_encoder.word_emb.weight.data.copy_(opt.pre_trained_ans_emb)
        model.decoder.word_emb.weight.data.copy_(opt.pre_trained_tgt_emb)
    
    if opt.proj_share_weight:
        weight = model.decoder.maxout(model.decoder.word_emb.weight.data)
        model.generator.weight.data.copy_(weight)

    return model, parameters_cnt


def build_model(opt, device, separate=-1, checkpoint=None):
    encoder = build_encoder(opt, separate=separate)
    decoder = build_decoder(opt, device)
    if opt.answer == 'enc':
        answer_encoder = build_encoder(opt, answer=True)
        model = OpenNQG(encoder, decoder, answer_encoder=answer_encoder)
    else:
        model = OpenNQG(encoder, decoder)
  
    model.generator = nn.Linear(opt.d_dec_model // opt.maxout_pool_size, opt.tgt_vocab_size, bias=False)

    model, parameters_cnt = initialize(model, opt)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    model = model.to(device)    
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus)
    
    return model, parameters_cnt
