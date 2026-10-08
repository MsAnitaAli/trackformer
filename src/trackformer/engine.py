# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
from collections import deque  # added for branch con_triploss_buffer

import logging
import math
import os
import sys
from typing import Iterable

import torch
from track import ex

from .datasets import get_coco_api_from_dataset
from .datasets.coco_eval import CocoEvaluator
from .datasets.panoptic_eval import PanopticEvaluator
from .models.detr_segmentation import DETRSegm
from .util import misc as utils
from .util.box_ops import box_iou
from .util.track_utils import evaluate_mot_accums
from .vis import vis_results


def make_results(outputs, targets, postprocessors, tracking, return_only_orig=True):
    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

    results = None
    if not return_only_orig:
        results = postprocessors['bbox'](outputs, target_sizes)
    results_orig = postprocessors['bbox'](outputs, orig_target_sizes)

    if 'segm' in postprocessors:
        results_orig = postprocessors['segm'](
            results_orig, outputs, orig_target_sizes, target_sizes)
        if not return_only_orig:
            results = postprocessors['segm'](
                results, outputs, target_sizes, target_sizes)

    if results is None:
        return results_orig, results

    for i, result in enumerate(results):
        target = targets[i]
        target_size = target_sizes[i].unsqueeze(dim=0)

        result['target'] = {}
        result['boxes'] = result['boxes'].cpu()

        # revert boxes for visualization
        for key in ['boxes', 'track_query_boxes']:
            if key in target:
                target[key] = postprocessors['bbox'].process_boxes(
                    target[key], target_size)[0].cpu()

        if tracking and 'prev_target' in target:
            if 'prev_prev_target' in target:
                target['prev_prev_target']['boxes'] = postprocessors['bbox'].process_boxes(
                    target['prev_prev_target']['boxes'],
                    target['prev_prev_target']['size'].unsqueeze(dim=0))[0].cpu()

            target['prev_target']['boxes'] = postprocessors['bbox'].process_boxes(
                target['prev_target']['boxes'],
                target['prev_target']['size'].unsqueeze(dim=0))[0].cpu()

            if 'track_query_match_ids' in target and len(target['track_query_match_ids']):
                track_queries_iou, _ = box_iou(
                    target['boxes'][target['track_query_match_ids'].to(target['boxes'].device)],
                    result['boxes'])
                # line modified to handle visdom data display issue

                box_ids = [box_id
                    for box_id, (is_track_query, is_fals_pos_track_query)
                    in enumerate(zip(target['track_queries_mask'], target['track_queries_fal_pos_mask']))
                    if is_track_query and not is_fals_pos_track_query]

                result['track_queries_with_id_iou'] = torch.diagonal(track_queries_iou[:, box_ids])

    return results_orig, results


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, visualizers: dict, args):

    vis_iter_metrics = None
    if visualizers:
        vis_iter_metrics = visualizers['iter_metrics']

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=vis_iter_metrics,
        debug=args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    # ── Temporal gap buffer (con_triploss_buffer branch) ──────────────────────
    # Buffer is initialised OUTSIDE the loop so it accumulates across iterations.
    # Each entry stores track_query_hs_embeds per batch item from one past
    # training step. After GAP_K steps the oldest entry (k steps ago) is used
    # as the hard positive for the contrastive loss instead of the adjacent frame.
    # Note: consecutive iterations may come from different sequences due to
    # dataloader shuffling; loss_contrastive handles size mismatches via K_min
    # truncation and falls back to adjacent-frame positives when sizes differ.
    GAP_K = 5
    hs_embed_buffer = deque(maxlen=GAP_K)
    # ─────────────────────────────────────────────────────────────────────────

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        # ── Inject gap embeddings from k iterations ago ───────────────────────
        # This runs BEFORE the model call so the criterion sees the gap key.
        # model.forward does not inspect track_query_hs_embeds_gap so it is
        # safe to add it to targets here.
        if len(hs_embed_buffer) == GAP_K:
            gap_entry = hs_embed_buffer[0]  # oldest entry in the buffer
            for b_idx, t in enumerate(targets):
                if (b_idx < len(gap_entry)
                        and gap_entry[b_idx] is not None):
                    t['track_query_hs_embeds_gap'] = gap_entry[b_idx].to(device)
        # ─────────────────────────────────────────────────────────────────────

        # in order to be able to modify targets inside the forward call we need
        # to pass it through as torch.nn.parallel.DistributedDataParallel only
        # passes copies
        outputs, targets, *_ = model(samples, targets)

        # ── Store current track_query_hs_embeds for future gap use ───────────
        # Detach and move to CPU to avoid holding GPU memory across k steps.
        current_entry = []
        for t in targets:
            tqhe = t.get('track_query_hs_embeds', None)
            current_entry.append(
                tqhe.detach().cpu() if tqhe is not None else None)
        hs_embed_buffer.append(current_entry)
        # ─────────────────────────────────────────────────────────────────────

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if visualizers and (i == 0 or not i % args.vis_and_log_interval):
            results_orig, results = make_results(
                outputs, targets, postprocessors, args.tracking, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.unmasked_tensor(0),
                results[0],
                targets[0],
                args.tracking)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device,
             output_dir: str, visualizers: dict, args, epoch: int = None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        debug=args.debug)
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    base_ds = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = tuple(k for k in ('bbox', 'segm') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 'Test:')):
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, *_ = model(samples, targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if visualizers and (i == 0 or not i % args.vis_and_log_interval):
            results_orig, results = make_results(
                outputs, targets, postprocessors, args.tracking, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.unmasked_tensor(0),
                results[0],
                targets[0],
                args.tracking)
        else:
            results_orig, _ = make_results(outputs, targets, postprocessors, args.tracking)

        if coco_evaluator is not None:
            results_orig = {
                target['image_id'].item(): output
                for target, output in zip(targets, results_orig)}
            coco_evaluator.update(results_orig)

        if panoptic_evaluator is not None:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for j, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[j]["image_id"] = image_id
                res_pano[j]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in coco_evaluator.coco_eval:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # TRACK EVAL
    if args.tracking and args.tracking_eval:
        stats['track_bbox'] = []

        ex.logger = logging.getLogger("submitit")

        seqs = data_loader.dataset.sequences
        seqs_per_rank = {i: [] for i in range(utils.get_world_size())}
        for i, seq in enumerate(seqs):
            rank = i % utils.get_world_size()
            seqs_per_rank[rank].append(seq)

        if args.debug:
            seqs_per_rank = {k: v[:1] for k, v in seqs_per_rank.items()}
            seqs = [s for ss in seqs_per_rank.values() for s in ss]

        dataset_name = seqs_per_rank[utils.get_rank()]
        if not dataset_name:
            dataset_name = seqs_per_rank[0]

        model_without_ddp = model
        if args.distributed:
            model_without_ddp = model.module

        if isinstance(model, DETRSegm):
            model_without_ddp = model_without_ddp.detr

        obj_detector_model = {
            'model': model_without_ddp,
            'post': postprocessors,
            'img_transform': args.img_transform}

        config_updates = {
            'seed': None,
            'dataset_name': dataset_name,
            'frame_range': data_loader.dataset.frame_range,
            'obj_detector_model': obj_detector_model}
        run = ex.run(config_updates=config_updates)

        mot_accums = utils.all_gather(run.result)[:len(seqs)]
        mot_accums = [item for sublist in mot_accums for item in sublist]

        eval_summary, eval_summary_str = evaluate_mot_accums(mot_accums, seqs)
        print(eval_summary_str)

        for metric in ['mota', 'idf1']:
            eval_m = eval_summary[metric]['OVERALL']
            stats['track_bbox'].append(eval_m)

    eval_stats = stats['coco_eval_bbox'][:3]
    if 'coco_eval_masks' in stats:
        eval_stats.extend(stats['coco_eval_masks'][:3])
    if 'track_bbox' in stats:
        eval_stats.extend(stats['track_bbox'])

    # VIS
    if visualizers:
        vis_epoch = visualizers['epoch_metrics']
        y_data = [stats[legend_name] for legend_name in vis_epoch.viz_opts['legend']]
        vis_epoch.plot(y_data, epoch)
        visualizers['epoch_eval'].plot(eval_stats, epoch)

    if args.debug:
        exit()

    return eval_stats, coco_evaluator
