import torch
import torch.nn as nn

from .position_embedding import PositionEmbeddingCoordsSine
from .transformer import (
    TransformerDecoder,
    TransformerDecoderCrossLayer,
    TransformerDecoderLayer,
)


class QueryDecoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,#
        attn_mask=False,
        pe=False,
        temperature=10000,
        pos_type="fourier",
        attn_mask_thresh=0.5,
        quant_grid_length=24,#
        grid_size=0.05,#
        rel_query=True, #
        rel_key=True, #
        rel_value=True #
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.refpoint_embed = nn.Embedding(num_query, 3)

        self.key_position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=True, pos_type=pos_type, d_pos=d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, hidden_dim,
                                        dropout, activation_fn, normalize_before=False)
        decoder_layer_cross = TransformerDecoderCrossLayer(d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, hidden_dim,
                                        dropout, activation_fn, normalize_before=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, decoder_layer_cross, num_layer, decoder_norm,
                                          return_intermediate=True,
                                          nhead=nhead,
                                          d_model=d_model,
                                          attn_mask_thresh=attn_mask_thresh)

        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.out_bbox = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        nn.init.constant_(self.out_bbox[-1].weight.data, 0)
        nn.init.constant_(self.out_bbox[-1].bias.data, 0)

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_offsets, input_ranges, ref_points):
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)
        pred_bboxes = self.out_bbox(query)
        for i, input_range in enumerate(input_ranges):
            min_xyz_i, max_xyz_i = input_range
            pred_bboxes[i] = ref_points[i] * (max_xyz_i - min_xyz_i) + min_xyz_i + pred_bboxes[i]
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_scores, pred_bboxes, pred_masks, attn_masks


    def forward_iter_pred(self, x, pos, batch_offsets, epoch,
                     non_boundary_sp_feats=None, non_boundary_sp_coords=None, non_boundary_mask=None,
                     boundary_sp_feats=None, boundary_sp_coords=None, boundary_mask=None):
        """
        x [B*M, inchannel]
        """

        B = len(batch_offsets) - 1
        d_model = self.d_model

        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        prediction_bboxes = []
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)

        if non_boundary_sp_feats is not None:
            non_boundary_sp_feats = self.input_proj(non_boundary_sp_feats)  # [B*M, 32] -> [B*M, 256]
        if boundary_sp_feats is not None:
            boundary_sp_feats = self.input_proj(boundary_sp_feats)  # [B*M, 32] -> [B*M, 256]

        query = self.refpoint_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (b, n, 3)
        num_queries = query.shape[1]

        query = query.permute(1,0,2).contiguous() #[num_queries, b, 3]
        lengths = batch_offsets[1:] - batch_offsets[:-1]
        max_length = lengths.max().item()
        inst_feats_batched = inst_feats.new_zeros(max_length, B, d_model)
        pos_batched = pos.new_zeros(max_length, B, d_model)
        coords_float_batched = pos.new_zeros(max_length, B, 3)
        key_padding_masks_batched = inst_feats.new_ones(B, max_length).bool()
        mask_feats_batched = mask_feats.new_zeros(max_length, B, d_model)
        input_ranges = []

        if non_boundary_sp_feats is not None:
            non_boundary_feats_batched = inst_feats.new_zeros(max_length, B, d_model)
            non_boundary_coords_batched = pos.new_zeros(max_length, B, 3)
            non_boundary_mask_batched = inst_feats.new_ones(B, max_length).bool()
        else:
            non_boundary_feats_batched = None
            non_boundary_coords_batched = None
            non_boundary_mask_batched = None

        if boundary_sp_feats is not None:
            boundary_feats_batched = inst_feats.new_zeros(max_length, B, d_model)
            boundary_coords_batched = pos.new_zeros(max_length, B, 3)
            boundary_mask_batched = inst_feats.new_ones(B, max_length).bool()
        else:
            boundary_feats_batched = None
            boundary_coords_batched = None
            boundary_mask_batched = None

        for i in range(B):
            start, end = batch_offsets[i], batch_offsets[i+1]
            length = end - start

            inst_feats_batched[:length, i, :] = inst_feats[start:end]

            pos_i = pos[start:end]
            coords_float_batched[:length, i, :] = pos_i

            pos_i_min, pos_i_max = pos_i.min(0)[0], pos_i.max(0)[0]
            pos_emb_i = self.key_position_embedding(pos_i.unsqueeze(0), num_channels=d_model, input_range=(pos_i_min.unsqueeze(0), pos_i_max.unsqueeze(0)))[0]
            pos_batched[:length, i, :] = pos_emb_i
            input_ranges.append((pos_i_min, pos_i_max))

            mask_feats_batched[:length, i, :] = mask_feats[start:end]
            key_padding_masks_batched[i, :length] = False

            if non_boundary_sp_feats is not None:
                non_boundary_feats_batched[:length, i, :] = non_boundary_sp_feats[start:end]
                non_boundary_coords_batched[:length, i, :] = non_boundary_sp_coords[start:end]
                non_boundary_mask_batched[i, :length] = ~non_boundary_mask[start:end]

            if boundary_sp_feats is not None:
                boundary_feats_batched[:length, i, :] = boundary_sp_feats[start:end]
                boundary_coords_batched[:length, i, :] = boundary_sp_coords[start:end]
                boundary_mask_batched[i, :length] = ~boundary_mask[start:end]

        intermediate_results, ref_points, mask_feats_batched, self_attn = self.decoder(tgt=query.new_zeros(num_queries, B, d_model),
            memory=inst_feats_batched,
            non_boundary_feats=non_boundary_feats_batched,
            non_boundary_coords=non_boundary_coords_batched,
            non_boundary_mask=non_boundary_mask_batched,
            boundary_feats=boundary_feats_batched,
            boundary_coords=boundary_coords_batched,
            boundary_mask=boundary_mask_batched,
            input_ranges=input_ranges,
            coords_float=coords_float_batched,
            mask_feats_batched=mask_feats_batched,
            lengths=lengths,
            memory_key_padding_mask=key_padding_masks_batched,
            pos=pos_batched,
            ref_points_unsigmoid=query, epoch=epoch, batch_offsets=batch_offsets)


        mask_feats_list = []
        for layer in range(len(mask_feats_batched)):
            for i in range(B):
              start, end = batch_offsets[i], batch_offsets[i+1]
              mask_feats[start:end] = mask_feats_batched[layer][:lengths[i], i, :]
            mask_feats_list.append(mask_feats.clone())
        for i in range(len(intermediate_results)):
            ouptut_i = intermediate_results[i].transpose(0,1)
            pred_labels, pred_scores, pred_bboxes, pred_masks, attn_masks = self.prediction_head(ouptut_i, mask_feats_list[i+1], batch_offsets, input_ranges, ref_points[i])
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_bboxes.append(pred_bboxes)
            prediction_masks.append(pred_masks)


        return {
            'queries':
            ouptut_i,
            'labels':
            prediction_labels[-1],
            'masks':
            prediction_masks[-1],
            'scores':
            prediction_scores[-1],
            'bboxes':
            prediction_bboxes[-1],
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c,
                'bboxes': d,
            } for a, b, c, d in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
                prediction_bboxes[:-1],
            )],
        }, mask_feats_list, self_attn

    def forward(self, x, pos, batch_offsets, epoch,
                non_boundary_sp_feats=None, non_boundary_sp_coords=None,non_boundary_mask=None,
                boundary_sp_feats=None, boundary_sp_coords=None, boundary_mask=None):
        return self.forward_iter_pred(x, pos, batch_offsets, epoch,
                                    non_boundary_sp_feats=non_boundary_sp_feats,
                                    non_boundary_sp_coords=non_boundary_sp_coords,
                                    non_boundary_mask=non_boundary_mask,
                                    boundary_sp_feats=boundary_sp_feats,
                                    boundary_sp_coords=boundary_sp_coords,
                                    boundary_mask=boundary_mask)