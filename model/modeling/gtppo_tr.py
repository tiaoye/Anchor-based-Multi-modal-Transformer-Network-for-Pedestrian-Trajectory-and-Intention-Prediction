
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn


from model.layers.loss import cvae_loss, focal_loss

from .tr_utils import TriangularCausalMask, ProbMask
from .tr_utils import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .tr_utils import Decoder, DecoderLayer, DecoderStack, SimpleDecoder
from .tr_utils import FullAttention, ProbAttention, AttentionLayer
from .tr_utils import DataEmbedding
from .tr_utils import gen_sineembed_for_position, build_mlps
from .tr_utils import batch_nms

class GTPPOTR(nn.Module):
    def __init__(self, cfg, dataset_name=None):
        super(GTPPOTR, self).__init__()
        self.cfg = copy.deepcopy(cfg)

        enc_in = dec_in = cfg.GLOBAL_INPUT_DIM
        c_out = cfg.DEC_OUTPUT_DIM
        self.pred_len = out_len = cfg.PRED_LEN
        dropout = cfg.DROPOUT
        self.d_model = d_model = cfg.GLOBAL_EMBED_SIZE
        self.K = cfg.K

        # config not in cfg
        d_ff = 256
        e_layers = 3
        d_layers = 2
        n_heads = 8
        factor = 5
        freq = 'h'
        activation='gelu'
        self.attn = attn = 'full'
        distil=False
        self.output_attention = output_attention = False
        embed='fixed'

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(d_model)
        )

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        # self.projection = nn.Linear(d_model, c_out, bias=True)

        self.motion_reg_heads, self.motion_cls_heads = self.build_motion_head(d_model, d_model, d_layers, self.pred_len)

        # intention
        self.intention_classifer = nn.Sequential(nn.Linear(d_model,
                                                        128),
                                                nn.ReLU(),
                                                nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 2)) if cfg.PRED_INTENTION else None
        self.focal_loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=2)

        # learnable anchors
        # self.intention_query = nn.Parameter(torch.randn(1,self.K, 4)) # (1,K, d_model)
        # self.intention_query_mlps = build_mlps(
        #         c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
        #     )
        self.intention_query = nn.Parameter(torch.randn(1,self.K, d_model)) # (1,K, d_model)

    def forward(self, input_x:torch.Tensor, 
                target_y=None, 
                neighbors_st=None, 
                adjacency=None, 
                z_mode=False, 
                cur_pos=None, 
                first_history_indices=None,
                intent_label=None):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        loss_dict = loss_dict =  {'loss_goal':torch.tensor(0.0).to(input_x), 'loss_traj':torch.tensor(0.0).to(input_x), 'loss_kld':torch.tensor(0.0).to(input_x)}

        enc_self_mask=None
        dec_self_mask=None
        dec_enc_mask=None
        x_mark_enc = x_mark_dec = None
        x_enc = input_x # (batch_size, segment_len, 4)
        
        # 1. encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # (batch_size, segment_len/4, d_model)

        # 2. decoder
        # intention_query = gen_sineembed_for_position(self.intention_query, self.d_model/2).view(self.K, self.d_model) # (K, d_model)
        # query = self.intention_query_mlps(intention_query).unsqueeze(0).repeat(input_x.size(0), 1, 1) # (batch_size, K, d_model)
        query = self.intention_query.repeat(input_x.size(0), 1, 1) # (batch_size, K, d_model)
        dec_layers_out = self.decoder(query, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, return_all_layers=True)[1] # (batch_size, K, d_model) * d_layers

        # 3. motion prediction
        cur_pos = input_x[:, None, -1, :] # (batch_size, 1, 4)
        best_idx = None
        for i in range(len(dec_layers_out)):
            query = dec_layers_out[i].reshape(input_x.size(0)*self.K, self.d_model) # (batch_size*K, d_model)
            pred_traj = self.motion_reg_heads[i](query).reshape(input_x.size(0), self.K, self.pred_len, 4)
            pred_traj = pred_traj + cur_pos.unsqueeze(1) # (batch_size, K, pred_len, 4)
            pred_traj = pred_traj.permute(0, 2, 1, 3) # (batch_size, pred_len, K, 4)
            pred_score = self.motion_cls_heads[i](query).reshape(input_x.size(0), self.K) # (batch_size, K)

            if target_y is None:
                continue
            _, loss_traj, new_best_idx = cvae_loss(None, pred_traj, target_y, select_metric='traj_rmse', best_of_many=True) # (1)
            # use the same idx for all layers
            if best_idx is None:
                best_idx = new_best_idx
            loss_cls = F.cross_entropy(pred_score, best_idx, reduction='none').mean() # (1)

            # use the place of loss_goal for loss_cls
            loss_dict['loss_goal'] += loss_cls
            loss_dict['loss_traj'] += loss_traj
            loss_dict[f'loss_layer{i}'] = loss_traj + loss_cls
            loss_dict[f'loss_layer{i}_cls'] = loss_cls
            loss_dict[f'loss_layer{i}_traj'] = loss_traj

        loss_dict['loss_goal'] /= len(dec_layers_out)
        loss_dict['loss_traj'] /= len(dec_layers_out)

        # 6. pred intention if required(new added)
        pred_intent = None
        if self.intention_classifer is not None:
            h_x = dec_layers_out[-1] # (batch_size, K, d_model)
            pred_intent = self.intention_classifer(h_x)# (batch_size, K, 2)
            if target_y is not None:
                pred_intent = pred_intent[torch.arange(input_x.size(0),device=input_x.device), best_idx] # (batch_size, 2)
                loss_dict['loss_intention'] = self.focal_loss_fn(pred_intent, intent_label)
            else:
                chosen_idx = torch.argmax(pred_score, dim=-1) # (batch_size)
                pred_intent = pred_intent[torch.arange(input_x.size(0),device=input_x.device), chosen_idx] # (batch_size, 2)
            pred_intent = torch.softmax(pred_intent, dim=-1) # (batch_size, 2)
        
        # NMS if not training
        # if not self.training:
        #     pred_traj = pred_traj.permute(0, 2, 1, 3) # (batch_size, K, pred_len, 4)
        #     pred_score = torch.softmax(pred_score, dim=-1) # (batch_size, K)
        #     pred_traj, pred_score, _ = batch_nms(pred_traj, pred_score, dist_thresh=0.05, num_ret_modes=20)
        #     pred_traj = pred_traj.permute(0, 2, 1, 3) # (batch_size, 20, pred_len, 4)
        return None, pred_traj, loss_dict, None, None, pred_intent

    def build_motion_head(self,in_channels, hidden_size, num_decoder_layers, num_future_frames):
        motion_reg_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, num_future_frames * 4], ret_before_act=True
        )
        motion_cls_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        return motion_reg_heads, motion_cls_heads