# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from .attention_rpe import MultiheadAttentionRPE
from .position_embedding import PositionEmbeddingCoordsSine
from .misc import Conv2dNormActivation
from timm.models.layers import trunc_normal_

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, decoder_layer_cross, num_layers, norm=None, return_intermediate=False, nhead=8, d_model=256, temperature=10000, pos_type="fourier", attn_mask_thresh=0.5):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers_cross = _get_clones(decoder_layer_cross, num_layers//3)
        self.num_layers = num_layers
        self.nhead = nhead
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        for layer_id in range(num_layers - 1):#
            self.layers[layer_id + 1].ca_qpos_proj = None

        self.position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=False, pos_type=pos_type, d_pos=d_model)
        self.key_position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=True, pos_type=pos_type, d_pos=d_model)
        self.attn_mask_thresh = attn_mask_thresh

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self._reset_parameters()

        self.bbox_embed = MLP(d_model, d_model, 3, 3)
        self.position_relation_embedding = PositionRelationEmbedding(16, self.nhead)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, tgt, memory, non_boundary_feats, non_boundary_coords, non_boundary_mask, boundary_feats, boundary_coords, boundary_mask,input_ranges, coords_float, mask_feats_batched, lengths,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ref_points_unsigmoid: Optional[Tensor] = None,
                epoch=0,
                batch_offsets=None):
        output = tgt
        attn_masks = None

        intermediate = []
        reference_points = ref_points_unsigmoid.sigmoid().transpose(0, 1)# (bsz, num_queries, 3)

        ref_points = [reference_points]

        input_ranges_mins, input_ranges_maxs = [], []
        for i in range(len(input_ranges)):
            pos_i_min, pos_i_max = input_ranges[i]
            input_ranges_mins.append(pos_i_min) #[3]
            input_ranges_maxs.append(pos_i_max)
        input_ranges_mins = torch.stack(input_ranges_mins, dim=0).unsqueeze(0) #[1, bsz, 3]
        input_ranges_maxs = torch.stack(input_ranges_maxs, dim=0).unsqueeze(0) #[1, bsz, 3]

        def _ensure_not_all_masked(key_padding_mask: Optional[Tensor]) -> Optional[Tensor]:
            if key_padding_mask is None:
                return None
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
            return key_padding_mask

        non_boundary_mask = _ensure_not_all_masked(non_boundary_mask)
        boundary_mask = _ensure_not_all_masked(boundary_mask)

        pos_non_boundary = None
        if non_boundary_coords is not None:
            pos_non_boundary = self.key_position_embedding(
                non_boundary_coords.transpose(0, 1),# [bsz, num_non_boundary, 3]
                num_channels=self.d_model,
                input_range=(input_ranges_mins.squeeze(0), input_ranges_maxs.squeeze(0)),
            ).transpose(0, 1)# [num_non_boundary, bsz, d_model]

        pos_boundary = None
        if boundary_coords is not None:
            pos_boundary = self.key_position_embedding(
                boundary_coords.transpose(0, 1),# [bsz, num_boundary, 3]
                num_channels=self.d_model,
                input_range=(input_ranges_mins.squeeze(0), input_ranges_maxs.squeeze(0)),
            ).transpose(0, 1)# [num_boundary, bsz, d_model]
        memory_list = [memory]
        self_attn = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :3].transpose(0, 1)

            reference_points_coords_float = torch.zeros_like(reference_points) #[batch_size, num_queries, 3]
            B = len(input_ranges)
            for b in range(B):
                pos_i_min, pos_i_max = input_ranges[b] #[3]
                reference_points_coords_float[b] = reference_points[b] * (pos_i_max - pos_i_min) + pos_i_min
            reference_points_coords_float = reference_points_coords_float.transpose(0, 1)

            query_sine_embed = self.position_embedding(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)# [num_queries, batch_size, d_model]


            tgt_mask = None

            output, attn1 = layer(
                output,
                memory,
                query_coords_float=reference_points_coords_float,
                key_coords_float=coords_float,# (max_length, batch_size, 3)
                tgt_mask=tgt_mask,
                memory_mask=attn_masks,
                tgt_key_padding_mask=tgt_key_padding_mask,# (batch_size, num_queries)
                memory_key_padding_mask=memory_key_padding_mask,# (batch_size, max_length)
                pos=pos,# (max_length, batch_size, d_model)
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
                non_boundary_memory=non_boundary_feats,
                non_boundary_key_coords_float=non_boundary_coords,
                non_boundary_memory_key_padding_mask=non_boundary_mask,
                non_boundary_pos=pos_non_boundary,
                boundary_memory=boundary_feats,
                boundary_key_coords_float=boundary_coords,
                boundary_memory_key_padding_mask=boundary_mask,
                boundary_pos=pos_boundary,
            )

            self_attn.append(attn1)
            if layer_id % 3 == 0:
                memory = self.layers_cross[layer_id % 3](memory, output, query_coords_float=coords_float, key_coords_float=reference_points_coords_float, tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=None,
                            pos=query_pos, query_pos=pos, query_sine_embed=pos,
                            is_first=(layer_id%3 == 0))

            memory_list.append(memory)
            output_norm = self.norm(output)
            if self.return_intermediate:
                intermediate.append(output_norm)
            # get mask
            pred_masks = torch.einsum('nbd,mbd->bnm', output_norm, memory)

            attn_mask_thresh = self.attn_mask_thresh
            attn_masks = (pred_masks.sigmoid() < attn_mask_thresh).bool() #[bsz, tgt_len, src_len]
            for b in range(lengths.shape[0]):
                length = lengths[b]
                attn_masks[b, (attn_masks[b, :, :length].sum(-1) == length)] = False
                attn_masks[b, :, length:] = True


            obj_center_offset = self.bbox_embed(output_norm) #[num_queries, bsz, 3]
            new_reference_points = obj_center * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins + obj_center_offset #[num_queries, bsz, 3]
            new_reference_points = (new_reference_points - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins) #[num_queries, bsz, 3]
            new_reference_points = new_reference_points.transpose(0,1) #[bsz, num_queries, 3]


            attn_masks = attn_masks.unsqueeze(1).expand(-1, self.nhead, -1, -1).contiguous().flatten(0,1)

            if layer_id != len(self.layers) - 1:
              ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()


        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [intermediate,ref_points,memory_list,self_attn]

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.ca_qcontent_proj_nb = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_nb = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_nb = nn.Linear(d_model, d_model)
        self.ca_v_proj_nb = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_nb = nn.Linear(d_model, d_model)
        self.cross_attn_nb = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.ca_qcontent_proj_bd = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_bd = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_bd = nn.Linear(d_model, d_model)
        self.ca_v_proj_bd = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_bd = nn.Linear(d_model, d_model)
        self.cross_attn_bd = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm2_nb = nn.LayerNorm(d_model)
        self.norm2_bd = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_nb = nn.Dropout(dropout)
        self.dropout2_bd = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.quant_grid_length = quant_grid_length
        self.grid_size = grid_size
        self.rel_query, self.rel_key, self.rel_value = rel_query, rel_key, rel_value

        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_query_table, std=.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_key_table, std=.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_value_table, std=.02)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, query_coords_float, key_coords_float,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     non_boundary_memory: Optional[Tensor] = None,
                     non_boundary_key_coords_float: Optional[Tensor] = None,
                     non_boundary_memory_key_padding_mask: Optional[Tensor] = None,
                     non_boundary_pos: Optional[Tensor] = None,
                     boundary_memory: Optional[Tensor] = None,
                     boundary_key_coords_float: Optional[Tensor] = None,
                     boundary_memory_key_padding_mask: Optional[Tensor] = None,
                     boundary_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2, attn1 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        def _cross_attn_once(
            tgt_in: Tensor,
            mem: Tensor,
            mem_coords: Tensor,
            mem_key_padding_mask: Optional[Tensor],
            mem_pos: Tensor,
            attn_mask_local: Optional[Tensor],
            qcontent_proj: nn.Linear,
            kcontent_proj: nn.Linear,
            v_proj: nn.Linear,
            kpos_proj: nn.Linear,
            qpos_sine_proj: nn.Linear,
            cross_attn: MultiheadAttentionRPE,
            dropout_layer: nn.Dropout,
            norm_layer: nn.LayerNorm,
            use_qpos: bool,
        ) -> Tensor:
            q_content_local = qcontent_proj(tgt_in)
            k_content_local = kcontent_proj(mem)
            v_local = v_proj(mem)

            num_q, bs_local, n_model_local = q_content_local.shape
            hw_local, _, _ = k_content_local.shape

            k_pos_local = kpos_proj(mem_pos)
            if use_qpos:
                q_pos_local = self.ca_qpos_proj(query_pos)
                q_local = q_content_local + q_pos_local
                k_local = k_content_local + k_pos_local

            else:
                q_local = q_content_local
                k_local = k_content_local

            q_local = q_local.view(num_q, bs_local, self.nhead, n_model_local // self.nhead)
            q_sine = qpos_sine_proj(query_sine_embed)
            q_sine = q_sine.view(num_q, bs_local, self.nhead, n_model_local // self.nhead)
            q_local = torch.cat([q_local, q_sine], dim=3).view(num_q, bs_local, n_model_local * 2)

            k_local = k_local.view(hw_local, bs_local, self.nhead, n_model_local // self.nhead)
            k_pos_local = k_pos_local.view(hw_local, bs_local, self.nhead, n_model_local // self.nhead)
            k_local = torch.cat([k_local, k_pos_local], dim=3).view(hw_local, bs_local, n_model_local * 2)

            rel_pos_local = query_coords_float.unsqueeze(1) - mem_coords.unsqueeze(0)
            rel_idx_local = torch.div(rel_pos_local, self.grid_size, rounding_mode='floor').long()
            rel_idx_local[rel_idx_local < -self.quant_grid_length] = -self.quant_grid_length
            rel_idx_local[rel_idx_local > self.quant_grid_length - 1] = self.quant_grid_length - 1
            rel_idx_local += self.quant_grid_length

            tgt2_local = cross_attn(
                query=q_local,
                key=k_local,
                value=v_local,
                attn_mask=attn_mask_local,
                key_padding_mask=mem_key_padding_mask,
                rel_idx=rel_idx_local,
                relative_pos_query_table=self.relative_pos_query_table if self.rel_query else None,
                relative_pos_key_table=self.relative_pos_key_table if self.rel_key else None,
                relative_pos_value_table=self.relative_pos_value_table if self.rel_value else None,
            )[0]

            tgt_out = tgt_in + dropout_layer(tgt2_local)
            tgt_out = norm_layer(tgt_out)
            return tgt_out

        tgt = _cross_attn_once(
            tgt,
            memory,
            key_coords_float,
            memory_key_padding_mask,
            pos,
            memory_mask,
            self.ca_qcontent_proj,
            self.ca_kcontent_proj,
            self.ca_v_proj,
            self.ca_kpos_proj,
            self.ca_qpos_sine_proj,
            self.cross_attn,
            self.dropout2,
            self.norm2,
            use_qpos=is_first,
        )

        if non_boundary_memory is not None and non_boundary_key_coords_float is not None and non_boundary_pos is not None:
            tgt = _cross_attn_once(
                tgt,
                non_boundary_memory,
                non_boundary_key_coords_float,
                non_boundary_memory_key_padding_mask,
                non_boundary_pos,
                None,
                self.ca_qcontent_proj_nb,
                self.ca_kcontent_proj_nb,
                self.ca_v_proj_nb,
                self.ca_kpos_proj_nb,
                self.ca_qpos_sine_proj_nb,
                self.cross_attn_nb,
                self.dropout2_nb,
                self.norm2_nb,
                use_qpos=False,
            )

        if boundary_memory is not None and boundary_key_coords_float is not None and boundary_pos is not None:
            tgt = _cross_attn_once(
                tgt,
                boundary_memory,
                boundary_key_coords_float,
                boundary_memory_key_padding_mask,
                boundary_pos,
                None,
                self.ca_qcontent_proj_bd,
                self.ca_kcontent_proj_bd,
                self.ca_v_proj_bd,
                self.ca_kpos_proj_bd,
                self.ca_qpos_sine_proj_bd,
                self.cross_attn_bd,
                self.dropout2_bd,
                self.norm2_bd,
                use_qpos=False,
            )
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn1


    def forward(self, tgt, memory, query_coords_float, key_coords_float,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                non_boundary_memory: Optional[Tensor] = None,
                non_boundary_key_coords_float: Optional[Tensor] = None,
                non_boundary_memory_key_padding_mask: Optional[Tensor] = None,
                non_boundary_pos: Optional[Tensor] = None,
                boundary_memory: Optional[Tensor] = None,
                boundary_key_coords_float: Optional[Tensor] = None,
                boundary_memory_key_padding_mask: Optional[Tensor] = None,
                boundary_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):

        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(
            tgt,
            memory,
            query_coords_float,
            key_coords_float,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            non_boundary_memory,
            non_boundary_key_coords_float,
            non_boundary_memory_key_padding_mask,
            non_boundary_pos,
            boundary_memory,
            boundary_key_coords_float,
            boundary_memory_key_padding_mask,
            boundary_pos,
            query_pos,
            query_sine_embed,
            is_first,
        )


class TransformerDecoderCrossLayer(nn.Module):

    def __init__(self, d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.quant_grid_length = quant_grid_length
        self.grid_size = grid_size
        self.rel_query, self.rel_key, self.rel_value = rel_query, rel_key, rel_value

        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_query_table, std=.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_key_table, std=.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_value_table, std=.02)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, query_coords_float, key_coords_float,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):


        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        q = q_content
        k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)


        # contextual relative position encoding
        # query_coords_float: [num_queries, B, 3]
        # key_coords_float: [max_length, B, 3]
        rel_pos = query_coords_float.unsqueeze(1) - key_coords_float.unsqueeze(0) #[num_queries, max_length, B, 3]
        rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode='floor').long()
        rel_idx[rel_idx < -self.quant_grid_length] = -self.quant_grid_length
        rel_idx[rel_idx > self.quant_grid_length - 1] = self.quant_grid_length - 1

        rel_idx += self.quant_grid_length
        assert (rel_idx >= 0).all()
        assert (rel_idx <= 2*self.quant_grid_length-1).all()

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   rel_idx=rel_idx,
                                   relative_pos_query_table=self.relative_pos_query_table if self.rel_query else None,
                                   relative_pos_key_table=self.relative_pos_key_table if self.rel_key else None,
                                   relative_pos_value_table=self.relative_pos_value_table if self.rel_value else None)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def forward(self, tgt, memory, query_coords_float, key_coords_float,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(tgt, memory, query_coords_float, key_coords_float, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([3, 3], -1)
    xy2, wh2 = tgt_boxes.split([3, 3], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 6]

    return pos_embed


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.,
        scale=100.,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = PositionEmbeddingCoordsSine(
            temperature=temperature,
            normalize=False,
            pos_type="fourier",
            d_pos=embed_dim*4,
            d_in=6)

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 6, "src_boxes much have 6 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 6, "tgt_boxes must have 6 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed.reshape(src_boxes.shape[0],-1,6)).reshape(src_boxes.shape[0],src_boxes.shape[1],src_boxes.shape[1],-1).permute(0, 3, 1, 2)# [batch_size, embed_dim*4, num_boxes1, num_boxes2] embed_dim=16 num_heads=8
        pos_embed = self.pos_proj(pos_embed)# [batch_size, num_heads, num_boxes1, num_boxes2]

        return pos_embed.clone()

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")