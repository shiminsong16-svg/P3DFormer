import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from p3dformer.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock, MLP
from .query_decoder import QueryDecoder
import numpy as np
from .boundary import BoundaryHead
from torch_scatter import scatter_mean

@gorilla.MODELS.register_module()
class P3DFormer(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,#
        return_blocks=True,#
        pool='mean',
        num_class=18,
        decoder=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
        boundary_thresh=0.45,
    ):
        super().__init__()
        self.boundary_thresh = boundary_thresh

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)#
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class

        self.mlp =  nn.Sequential(nn.Linear(2*media, media), nn.ReLU(), nn.Linear(media, media))
        self.pooling_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.pooling_linear1 = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.coords_linear = MLP(3, media, norm_fn=norm_fn, num_layers=3)

        # sbm head
        self.bsm_head = BoundaryHead(
            in_channels=media,
            out_channels=1,
            hidden_dim=128,
            dropout=[0.3, 0.3]
        )

        self.fg_bg_head = MLP(media, 1, norm_fn=norm_fn, num_layers=3)

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        self.epoch = 0
        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:#
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):#
        super(P3DFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='predict'):
        if mode == 'predict':
            return self.predict(**batch)
        raise ValueError("This review-stage release supports inference only.")

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts, superpoints, coords_float,
                batch_offsets, sp_instance_labels, sp_semantic_labels):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        sp_coords1 = scatter_mean(coords_float, superpoints, dim=0, dim_size=batch_offsets[-1])  # (B*M, media)
        sp_feats, point_feats, fg_logits = self.extract_feat(input, superpoints, p2v_map, sp_coords1, coords_float)

        point_batch_offsets = torch.zeros(batch_size + 1, device=coords_float.device, dtype=torch.int32)
        cur = 0
        for i, inst in enumerate(insts):
            num_points = getattr(inst, 'num_points', coords_float.shape[0] // batch_size)
            point_batch_offsets[i] = cur
            cur += num_points
        point_batch_offsets[batch_size] = cur

        all_sp_feats = sp_feats  # [B*M, C]
        all_sp_coords = sp_coords1  # [B*M, 3]

        boundary_logits = self.bsm_head(point_feats)

        boundary_logits = boundary_logits[:, 0]
        boundary_prob = torch.sigmoid(boundary_logits)

        predict_is_boundary = (boundary_prob > self.boundary_thresh)
        predict_is_non_boundary = ~predict_is_boundary

        non_boundary_sp_feats, non_boundary_sp_coords, non_boundary_mask = self.create_boundary_filtered_superpoints(
            coords_float, point_feats, superpoints, predict_is_non_boundary, point_batch_offsets, batch_offsets, fg_logits
        )

        boundary_sp_feats, boundary_sp_coords, boundary_mask = self.create_boundary_filtered_superpoints(
            coords_float, point_feats, superpoints, predict_is_boundary, point_batch_offsets, batch_offsets, fg_logits
        )

        out, _, self_attn = self.decoder(
            all_sp_feats, all_sp_coords, batch_offsets, self.epoch,
            non_boundary_sp_feats=non_boundary_sp_feats,
            non_boundary_sp_coords=non_boundary_sp_coords,
            non_boundary_mask=non_boundary_mask,
            boundary_sp_feats=boundary_sp_feats,
            boundary_sp_coords=boundary_sp_coords,
            boundary_mask=boundary_mask
        )
        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)

        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        nms_score = scores.max(-1)[0].squeeze()
        proposals_pred_f = (pred_masks[0]>0).float()
        intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
        proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
        nms_score[proposals_pointnum==0] = 0
        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection+1e-6)
        pick_idxs = non_max_suppression(cross_ious.cpu().numpy(),nms_score.detach().cpu().numpy(), 0.75)

        pred_labels = pred_labels[:,pick_idxs]
        pred_masks[0] = pred_masks[0][pick_idxs]
        scores = scores[pick_idxs]
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(pred_labels.shape[1], 1).flatten(0, 1)

        self.test_cfg.topk_insts = min(self.test_cfg.topk_insts, scores.flatten(0, 1).shape[0])
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred = ((mask_pred > 0)).float()   # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = round(score_pred[i], 1)
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def create_boundary_filtered_superpoints(
        self,
        coords_float,
        feats,
        superpoints,
        point_mask,
        point_batch_offsets,
        sp_batch_offsets,
        sp_fg_logits
    ):
        device = feats.device
        B = len(point_batch_offsets) - 1
        C = feats.shape[1]

        sp_fg_probs = torch.sigmoid(sp_fg_logits) # [B*M, 1]

        out_feats = []
        out_coords = []
        out_masks = []

        for b in range(B):
            p_start, p_end = point_batch_offsets[b], point_batch_offsets[b+1]
            sp_start, sp_end = sp_batch_offsets[b], sp_batch_offsets[b+1]
            num_sp = sp_end - sp_start

            mask = point_mask[p_start:p_end]

            if mask.sum() == 0:
                sp_feats = torch.zeros(num_sp, C, device=device)
                sp_coords = torch.zeros(num_sp, 3, device=device)
                sp_mask = torch.zeros(num_sp, dtype=torch.bool, device=device)
            else:
                sp_ids = superpoints[p_start:p_end][mask] - sp_start
                feats_b = feats[p_start:p_end][mask]
                coords_b = coords_float[p_start:p_end][mask]

                sp_feats = scatter_mean(
                    feats_b, sp_ids, dim=0, dim_size=num_sp
                )

                batch_fg_probs = sp_fg_probs[sp_start:sp_end] # [num_sp, 1]
                sp_feats = sp_feats * batch_fg_probs
                # ---------------------------

                sp_coords = scatter_mean(
                    coords_b, sp_ids, dim=0, dim_size=num_sp
                )

                sp_mask = torch.zeros(num_sp, dtype=torch.bool, device=device)
                unique_sp_ids = torch.unique(sp_ids)
                sp_mask[unique_sp_ids] = True

            out_feats.append(sp_feats)
            out_coords.append(sp_coords)
            out_masks.append(sp_mask)

        return (
            torch.cat(out_feats),   # [B*M, C]
            torch.cat(out_coords),  # [B*M, 3]
            torch.cat(out_masks)
        )

    def extract_feat(self, x, superpoints, v2p_map, sp_coords, coords_float):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        x_origin = x.clone()
        x = scatter_mean(x_origin, superpoints, dim=0)  # (B*M, media)

        fg_logits = self.fg_bg_head(x) # [B*M, 1]
        fg_prob = torch.sigmoid(fg_logits)

        sp_feats_suppressed = x * fg_prob
        # ---------------------------

        return sp_feats_suppressed, x_origin, fg_logits


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where((iou > threshold))[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
