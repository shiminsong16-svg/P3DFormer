import argparse
import gorilla
import torch
from tqdm import tqdm
import os
import time
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from p3dformer.dataset import build_dataloader, build_dataset
from p3dformer.evaluation import ScanNetEval
from p3dformer.model import P3DFormer
from p3dformer.utils import get_root_logger, save_gt_instances, save_pred_instances
import random
import numpy as np

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser('P3DFormer evaluation')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    set_deterministic(cfg.test.seed)
    logger = get_root_logger()

    model_cfg = cfg.model.copy()
    model_name = model_cfg.pop("name", "P3DFormer")
    if model_name != "P3DFormer":
        raise ValueError(f"Unsupported model in this review release: {model_name}")
    model = P3DFormer(**model_cfg).cuda()

    data_dir = os.path.join(cfg.data.test.data_root, cfg.data.test.prefix)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f'Processed dataset split not found: {data_dir}. '
            'Update data.test.data_root in the config or prepare ScanNetV2 first.'
        )

    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []
    sem_labels, ins_labels = [], []
    coords = []
    progress_bar = tqdm(total=len(dataloader))
    pure_inf_time = 0
    with torch.no_grad():
        model.eval()
        for b, batch in enumerate(dataloader):
            batch.pop("batch_points_offsets", "")
            xyz, _, _, semantic_label, instance_label, _ = dataset.load(dataset.filenames[b])

            if cfg.data.test.type == "scannetv2":
                semantic_label[semantic_label != -100] -= 2
                semantic_label[(semantic_label == -1) | (semantic_label == -2)] = -100
            torch.cuda.synchronize()
            start_time = time.perf_counter()
                  
            result = model(batch, mode='predict')

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            pure_inf_time += elapsed
            results.append(result)
            sem_labels.append(semantic_label)
            ins_labels.append(instance_label)
            coords.append(xyz)
            progress_bar.update()
        progress_bar.close()
    for res in results:

        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not cfg.data.test.prefix == 'test':
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(pred_insts, gt_insts)
    # save output
    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
