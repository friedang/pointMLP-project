"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from util.S3DISDataLoader import S3DISDataset,ScannetDatasetWholeScene
import model as models

import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

g_class2label = {cls: i for i,cls in enumerate(classes)}
g_class2color = {'ceiling':	[0,255,0],
                 'floor':	[0,0,255],
                 'wall':	[0,255,255],
                 'beam':        [255,255,0],
                 'column':      [255,0,255],
                 'window':      [100,100,255],
                 'door':        [200,200,100],
                 'table':       [170,120,200],
                 'chair':       [255,0,0],
                 'sofa':        [200,100,100],
                 'bookcase':    [10,200,100],
                 'board':       [200,200,200],
                 'clutter':     [50,50,50]}
g_label2color = {classes.index(cls): g_class2color[cls] for cls in classes}


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point number [default: 4096]')
    # parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--id', type=int, default='8')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'figures'
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # Dataloader

    #root = 'data/custom_s3/'
    root = "/cluster/51/pointMLP_data/data/custom_s3/"
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=5, block_points=2048)
    num_batches = len(TEST_DATASET_WHOLE_SCENE)


    whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[args.id]
    whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[args.id]

    print("WHOLE SCENE DATA SHAPE",whole_scene_data.shape)
    print("WHOLE SCENE LABEL SHAPE",whole_scene_label.shape)

    print("===> Create model...")
    num_part = 13
    device = torch.device("cuda") # if args.cuda else "cpu")
    model = models.__dict__["pointMLP"](num_part).to(device)
    model.eval()
    print("===> Load checkpoint...")



    from collections import OrderedDict
    # state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
    #                         map_location=torch.device('cpu'))['model']
    state_dict = torch.load("./checkpoints/pointMLP_introduceUpsampling/best_acc_model.pth",
                            map_location=torch.device('cpu'))['model']

    # state_dict = torch.load("checkpoints/pointMLP_demo1/best_insiou_model.pth",
    #                         map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict, strict=False)

    # Evaluation
    batch_idx = args.id
    scene_id = TEST_DATASET_WHOLE_SCENE.file_list
    scene_id = [x[:-4] for x in scene_id]

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    log_string('---- EVALUATION WHOLE SCENE----')

    with torch.no_grad():
        print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
        total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
        if args.visual:
            fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
            fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        for _ in tqdm(range(args.num_votes), total=args.num_votes):
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

                batch_data[:, :, 3:6] /= 1.0
                whole_points = batch_data[:,:3]
                whole_color = batch_data[:,3:]
                whole_target = whole_scene_label
                whole_normal = whole_points / TEST_DATASET_WHOLE_SCENE.room_coord_max[5]

                whole_points = torch.tensor(whole_points).unsqueeze(dim=0)
                whole_color = torch.tensor(whole_color).unsqueeze(dim=0)
                whole_target = torch.tensor(whole_target).unsqueeze(dim=0)
                whole_normal = torch.tensor(whole_normal).unsqueeze(dim=0)

                whole_points = whole_points.transpose(2,1)
                whole_color = whole_color.transpose(2, 1)
                whole_normal = whole_normal.transpose(2, 1)
                whole_points, whole_target, whole_color, whole_normal = whole_points.cuda(non_blocking=True), whole_target.cuda(non_blocking=True), whole_color.cuda(non_blocking=True) ,whole_normal.cuda(non_blocking=True)


                seg_pred, _ = model(whole_points, whole_normal, whole_color)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                            batch_pred_label[0:real_batch_size, ...],
                                            batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

        for l in range(NUM_CLASSES):
            total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
            total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
            total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
            total_seen_class[l] += total_seen_class_tmp[l]
            total_correct_class[l] += total_correct_class_tmp[l]
            total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

        iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
        print(iou_map)
        arr = np.array(total_seen_class_tmp)
        tmp_iou = np.mean(iou_map[arr != 0])
        log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
        print('----------------------------')

        filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
        with open(filename, 'w') as pl_save:
            for i in pred_label:
                pl_save.write(str(int(i)) + '\n')
            pl_save.close()
        for i in range(whole_scene_label.shape[0]):
            color = g_label2color[pred_label[i]]
            color_gt = g_label2color[whole_scene_label[i]]
            if args.visual:
                fout.write('v %f %f %f %d %d %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                    color[2]))
                fout_gt.write(
                    'v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                        color_gt[1], color_gt[2]))
        if args.visual:
            fout.close()
            fout_gt.close()

    IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %s, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
            total_correct_class[l] / float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    log_string('eval point avg class IoU: %f' % np.mean(IoU))
    log_string('eval whole scene point avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
    log_string('eval whole scene point accuracy: %f' % (
            np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
