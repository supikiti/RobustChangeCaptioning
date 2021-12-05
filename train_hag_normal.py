import gc
import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tensorboardX import SummaryWriter

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import ChangeDetectorDoubleAttDyn, AddSpatialInfo
from models.dynamic_speaker import DynamicSpeaker
from models.dynamic_graph_speaker import DynamicGraphSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss

from utils.vis_utils import visualize_att


def parser():
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--img_feat_base_path', default="hag_data/output_true_all_images")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--entropy_weight', type=float, default=0.0001)
    parser.add_argument('--visualize_every', type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="experiments/logs")
    args = parser.parse_args()
    return args


def get_img_id(path):
    stem = Path(path).stem
    return "_".join(stem.split("_")[:-1])


def write_as_tensorboard(writer, data, iteration, type="train"):
    writer.add_scalars(type, data, iteration)
    return writer


def main(args):
    merge_cfg_from_file(args.cfg)
    assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

    writer = SummaryWriter(log_dir=f"{args.log_dir}/{cfg.exp_name}")

    # Device configuration
    use_cuda = torch.cuda.is_available()
    gpu_ids = cfg.gpu_id
    torch.backends.cudnn.enabled  = True
    default_gpu_device = gpu_ids[0]
    torch.cuda.set_device(default_gpu_device)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Experiment configuration
    exp_dir = cfg.exp_dir
    exp_name = cfg.exp_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    output_dir = os.path.join(exp_dir, exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg_file_save = os.path.join(output_dir, 'cfg.json')
    json.dump(cfg, open(cfg_file_save, 'w'))

    sample_dir = os.path.join(output_dir, 'eval_gen_samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sample_subdir_format = '%s_samples_%d'

    sent_dir = os.path.join(output_dir, 'eval_sents')
    if not os.path.exists(sent_dir):
        os.makedirs(sent_dir)
    sent_subdir_format = '%s_sents_%d'

    snapshot_dir = os.path.join(output_dir, 'snapshots')
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    snapshot_file_format = '%s_checkpoint_%d.pt'

    train_logger = Logger(cfg, output_dir, is_train=True)
    val_logger = Logger(cfg, output_dir, is_train=False)

    # Create model
    change_detector = ChangeDetectorDoubleAttDyn(cfg)
    change_detector.to(device)

    speaker = DynamicSpeaker(cfg)
    speaker.to(device)

    spatial_info = AddSpatialInfo()
    spatial_info.to(device)

    with open(os.path.join(output_dir, 'model_print'), 'w') as f:
        print(change_detector, file=f)
        print(speaker, file=f)
        print(spatial_info, file=f)

    # Data loading part
    train_dataset, train_loader = create_dataset(cfg, 'train', args.img_feat_base_path)
    val_dataset, val_loader = create_dataset(cfg, 'dev', args.img_feat_base_path)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    # Define loss function and optimizer
    lang_criterion = LanguageModelCriterion().to(device)
    entropy_criterion = EntropyLoss().to(device)
    all_params = list(change_detector.parameters()) +\
        list(speaker.parameters())
    optimizer = build_optimizer(all_params, cfg)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.train.optim.step_size,
        gamma=cfg.train.optim.gamma)

    # Train loop
    t = 0
    epoch = 0

    set_mode('train', [change_detector, speaker])
    ss_prob = speaker.ss_prob

    while t < cfg.train.max_iter:
        epoch += 1
        print('Starting epoch %d' % epoch)
        lr_scheduler.step()
        speaker_loss_avg = AverageMeter()
        total_loss_avg = AverageMeter()
        if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
            frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
            ss_prob_prev = ss_prob
            ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                        cfg.train.scheduled_sampling_max_prob)
            speaker.ss_prob = ss_prob
            if ss_prob_prev != ss_prob:
                print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))

        for i, batch in enumerate(train_loader):
            iter_start_time = time.time()

            img_1_feature, img_2_feature, cap_labels, \
                _, cap_masks, _, _, _ = batch

            batch_size = img_1_feature.size(0)
            cap_masks = cap_masks.float()

            img_1_feature, img_2_feature = img_1_feature.to(device), img_2_feature.to(device)
            cap_labels = cap_labels.to(device)
            cap_masks = cap_masks.to(device)

            img_1_feature, img_2_feature = spatial_info(img_1_feature), spatial_info(img_2_feature)

            optimizer.zero_grad()

            chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
            chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(img_1_feature, img_2_feature)

            speaker_output_pos = speaker._forward(chg_pos_feat_bef,
                                                chg_pos_feat_aft,
                                                chg_pos_feat_diff,
                                                cap_labels)

            dynamic_atts = speaker.get_module_weights() # (batch, seq_len, 3)

            speaker_loss = 1.0 * lang_criterion(speaker_output_pos[:, :-1, :], cap_labels[:, 1:], cap_masks[:, 1:])
            speaker_loss_val = speaker_loss.item()

            entropy_loss = -args.entropy_weight * entropy_criterion(dynamic_atts[:, :-1, :], cap_masks[:, 1:])
            att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum()) / (2 * batch_size)
            total_loss = speaker_loss + 2.5e-03 * att_sum + entropy_loss
            total_loss_val = total_loss.item()

            speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
            total_loss_avg.update(total_loss_val, 2 * batch_size)

            stats = {}
            stats['entropy_loss'] = entropy_loss.item()
            stats['speaker_loss'] = speaker_loss_val
            stats['avg_speaker_loss'] = speaker_loss_avg.avg
            stats['total_loss'] = total_loss_val
            stats['avg_total_loss'] = total_loss_avg.avg

            writer = write_as_tensorboard(writer, stats, t, type="train")

            total_loss.backward()
            optimizer.step()

            iter_end_time = time.time() - iter_start_time

            t += 1

            del speaker_output_pos
            gc.collect()

            if t % cfg.train.log_interval == 0:
                train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
                train_logger.plot_current_stats(
                    epoch,
                    float(i * batch_size) / train_size, stats, 'loss')

            if t % cfg.train.snapshot_interval == 0:
                speaker_state = speaker.state_dict()
                chg_det_state = change_detector.state_dict()
                checkpoint = {
                    'change_detector_state': chg_det_state,
                    'speaker_state': speaker_state,
                    'model_cfg': cfg
                }
                save_path = os.path.join(snapshot_dir,
                                        snapshot_file_format % (exp_name, t))
                save_checkpoint(checkpoint, save_path)

                print('Running eval at iter %d' % t)
                set_mode('eval', [change_detector, speaker])
                with torch.no_grad():
                    test_iter_start_time = time.time()

                    idx_to_word = train_dataset.get_idx_to_word()
                    idx_to_scene = train_dataset.get_idx_to_scene()

                    if args.visualize:
                        sample_subdir_path = sample_subdir_format % (exp_name, t)
                        sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                        if not os.path.exists(sample_save_dir):
                            os.makedirs(sample_save_dir)
                    sent_subdir_path = sent_subdir_format % (exp_name, t)
                    sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                    if not os.path.exists(sent_save_dir):
                        os.makedirs(sent_save_dir)

                    result_sents_pos = {}
                    val_speaker_loss_avg = AverageMeter()
                    val_total_loss_avg = AverageMeter()

                    for val_i, val_batch in enumerate(val_loader):

                        (img_1_feature, img_2_feature, cap_labels, _, 
                            cap_masks, _, img_1_path, img_2_path) = val_batch

                        val_batch_size = img_1_feature.size(0)
                        cap_masks = cap_masks.float()

                        img_1_feature, img_2_feature = img_1_feature.to(device), img_2_feature.to(device)
                        cap_labels, cap_masks = cap_labels.to(device), cap_masks.to(device)

                        img_1_feature, img_2_feature = spatial_info(img_1_feature), spatial_info(img_2_feature)

                        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
                        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(img_1_feature, img_2_feature)

                        #del img_1_feature, img_2_feature, chg_pos_logits, cap_masks, scene_masks
                        #gc.collect()

                        #print(chg_pos_feat_bef.shape, chg_pos_feat_aft.shape, chg_pos_feat_diff.shape, cap_labels.shape, scene_labels.shape)

                        speaker_output_pos = speaker._forward(chg_pos_feat_bef,
                                                                chg_pos_feat_aft,
                                                                chg_pos_feat_diff,
                                                                cap_labels)

                        pos_dynamic_atts = speaker.get_module_weights() # (batch, seq_len, 3)

                        val_speaker_loss = 1.0 * lang_criterion(speaker_output_pos[:, :-1, :], cap_labels[:, 1:], cap_masks[:, 1:])
                        val_speaker_loss_val = speaker_loss.item()

                        val_entropy_loss = -args.entropy_weight * entropy_criterion(pos_dynamic_atts[:, :-1, :], cap_masks[:, 1:])
                        val_att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum()) / (2 * batch_size)
                        val_total_loss = val_speaker_loss + 2.5e-03 * val_att_sum + val_entropy_loss
                        val_total_loss_val = val_total_loss.item()

                        val_speaker_loss_avg.update(val_speaker_loss_val, 2 * batch_size)
                        val_total_loss_avg.update(val_total_loss_val, 2 * batch_size)

                        stats = {}
                        stats['entropy_loss'] = entropy_loss.item()
                        stats['speaker_loss'] = val_speaker_loss_val
                        stats['avg_speaker_loss'] = val_speaker_loss_avg.avg
                        stats['total_loss'] = val_total_loss_val
                        stats['avg_total_loss'] = val_total_loss_avg.avg

                        writer = write_as_tensorboard(writer, stats, t, type="val")

                        speaker_output_pos_for_de, _ = speaker._sample(chg_pos_feat_bef,
                                                                chg_pos_feat_aft,
                                                                chg_pos_feat_diff,
                                                                cap_labels, cfg)

                        gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos_for_de)

                        chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
                        chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()

                        dummy = np.ones_like(chg_pos_att_bef)

                        gts = decode_sequence(idx_to_word, cap_labels)
                        gen_sent_length = cap_labels.size(1)

                        pos_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy() # (batch, seq_len, 3)

                        for val_j in range(speaker_output_pos.size(0)):
                            if args.visualize and val_j % args.visualize_every == 0:
                                visualize_att(img_1_path[val_j], img_2_path[val_j],
                                            chg_pos_att_bef[val_j], dummy[val_j], chg_pos_att_aft[val_j],
                                            pos_dynamic_atts[val_j], gen_sent_length, \
                                            gen_sents_pos[val_j], gts[val_j],
                                            sample_save_dir, 'sc_')

                            sent_pos = gen_sents_pos[val_j]
                            image_id = get_img_id(img_1_path[val_j])
                            result_sents_pos[image_id] = sent_pos
                            message = '%s results:\n' % image_id
                            message += '\t' + sent_pos + '\n'
                            message += '----------<GROUND TRUTHS>----------\n'
                            message += gts[val_j] + '\n'
                            message += '===================================\n'
                            print(message)

                    test_iter_end_time = time.time() - test_iter_start_time
                    result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                    coco_gen_format_save(result_sents_pos, result_save_path_pos)

                set_mode('train', [change_detector, speaker])
                writer.export_scalars_to_json(f"{exp_dir}/all_scalars.json")


if __name__ == "__main__":
    args = parser()
    main(args)