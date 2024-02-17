"""
Plot the parts.
python plot_ptseg.py --model model31G --exp_name demo1 --id 1
"""
from __future__ import print_function
import os
import argparse
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
from util.S3DISDataLoader import S3DISDataset,ScannetDatasetWholeScene
import pyntcloud
import pandas as pd
#import open3d as o3d
from sklearn.metrics import accuracy_score

classes_str = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

rgb_values = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Lime Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Green
    (0, 0, 128),      # Navy
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192)   # Silver
]

def test(args):
    # Dataloader
    
    #root = 'data/custom_s3/'
    root = "/cluster/51/pointMLP_data/data/custom_s3/"
    test_data = ScannetDatasetWholeScene(root, split='test', test_area=5, block_points=2048)
    num_batches = len(test_data)


    whole_scene_data = test_data.scene_points_list[args.id]
    whole_scene_label = test_data.semantic_labels_list[args.id]

    print("WHOLE SCENE DATA SHAPE",whole_scene_data.shape)
    print("WHOLE SCENE LABEL SHAPE",whole_scene_label.shape)

    num_points = whole_scene_data.shape[0]

    color_target_list = []
    for i in range(num_points):
            color_target = rgb_values[int(whole_scene_label[i])]
            r_target, g_target, b_target = color_target
            color_target_list.append([r_target, g_target, b_target])

    color_target_list = np.array(color_target_list)
    seg_colored_points = np.hstack((whole_scene_data[:, :3], color_target_list))
    fout = open( f"figures/{args.id}_target-whole-seg-color.obj", 'w')
    for k in range(num_points):
        fout.write('v %f %f %f %d %d %d\n' % (seg_colored_points[k][0],seg_colored_points[k][1],seg_colored_points[k][2],seg_colored_points[k][3],seg_colored_points[k][4],seg_colored_points[k][5]))

    print("===> Create model...")
    num_part = 13
    device = torch.device("cuda" if args.cuda else "cpu")
    model = models.__dict__[args.model](num_part).to(device)
    print("===> Load checkpoint...")



    from collections import OrderedDict
    # state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
    #                         map_location=torch.device('cpu'))['model']
    state_dict = torch.load("fred/best_acc_model.pth",
                            map_location=torch.device('cpu'))['model']
    
    # state_dict = torch.load("checkpoints/pointMLP_demo1/best_insiou_model.pth",
    #                         map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict, strict=False)

 
    print("===> Start evaluate...")
    #model.eval()
    num_classes = 13
    whole_points = whole_scene_data[:,:3]
    whole_color = whole_scene_data[:,3:]
    whole_target = whole_scene_label

    #print("check max coord",test_data.room_coord_max[5])

    whole_normal = whole_points / test_data.room_coord_max[5]
   # normal_x, normal_y, normal_z = normal[0],normal[1], normal[2]

    whole_points = torch.tensor(whole_points).unsqueeze(dim=0)
    whole_color = torch.tensor(whole_color).unsqueeze(dim=0)
    whole_target = torch.tensor(whole_target).unsqueeze(dim=0)
    whole_normal = torch.tensor(whole_normal).unsqueeze(dim=0)

    whole_points, whole_target, whole_color, whole_normal  = Variable(whole_points.float()), Variable(whole_target.long()), Variable(whole_color.float()),  Variable(whole_normal.float())

    whole_points = whole_points.transpose(2,1)
    whole_color = whole_color.transpose(2, 1)
    whole_normal = whole_normal.transpose(2, 1)
    whole_points, whole_target, whole_color, whole_normal = whole_points.cuda(non_blocking=True), whole_target.cuda(non_blocking=True), whole_color.cuda(non_blocking=True) ,whole_normal.cuda(non_blocking=True)

    print( "BEFORE MODELL POINTS SHAPE", whole_points.shape)
    print( "BEFORE MODEL COLOR SHAPE", whole_color.shape)
    print( "BEFORE MODEL LABEL SHAPE", whole_target.shape)


    # with torch.no_grad():
    #         predict = model(whole_points, whole_normal, whole_color)
    #         #predict = model(whole_points, whole_normal)

    
    # Define the chunk size
    chunk_size = 2048

    # Get the total number of chunks
    total_chunks = whole_points.size(2) // chunk_size

    #Initialize an empty tensor to store predictions
    predictions = torch.zeros_like(whole_target)

    print("bos pred shape", predictions.shape)
    # Loop through chunks and make predictions
    print("chunk num",total_chunks)
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        # Extract a chunk of the input tensor
        input_chunk_point = whole_points[:, :, start_idx:end_idx]
        input_chunk_normal = whole_normal[:, :, start_idx:end_idx]
        input_chunk_color = whole_color[:, :, start_idx:end_idx]
        # print("chunkkk what point ", input_chunk_point.shape)
        # print("chunkkk what normal", input_chunk_normal.shape)
        # print("chunkkk what color", input_chunk_color.shape)
        with torch.no_grad():
            predict = model(input_chunk_point, input_chunk_normal, input_chunk_color)
       # print("predict shapeee",predict.shape)
        predict = predict.max(dim=-1)[1]
        print(predict)
        #print("predict shapeee",predict.shape)

        predictions[:, start_idx:end_idx] = predict
    
    
    #predict = predict.max(dim=-1)[1]
    #predict = predict.squeeze(dim=0).cpu().data.numpy()  # 2048
    predictions = predictions.squeeze(dim=0).cpu().data.numpy()  # 2048
    print("PRED LAST HALI!!!",predictions[0])
    
    whole_target = whole_target.squeeze(dim=0).cpu().data.numpy()   # 2048

    accuracy = np.mean(whole_target == predictions)

    # Print the accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")

    accuracy = accuracy_score(whole_target, predictions)

    # Print the accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")
    #import pdb; pdb.set_trace()
    whole_points = whole_points.transpose(2, 1).squeeze(dim=0).cpu().data.numpy()
    whole_normal = whole_normal.transpose(2, 1).squeeze(dim=0).cpu().data.numpy()

    print( "AFTER MODELL PREDICT SHAPE", predictions.shape)
    print( "AFTER MODELL POINTS SHAPE", whole_points.shape)
    print( "AFTER COLOR SHAPE", whole_color.shape)
    print( "AFTER LABEL SHAPE", whole_target.shape)
    print( "AFTER NORMAL SHAPE", whole_normal.shape)

    #np.savetxt(f"figures/{args.id}-point.txt", points)
    
    #################  CREATE TXT FILES WITH XYZ RGB #################
    print("UNIQUE PREDICTION CLASSES !!!",np.unique(predictions))

    color_predict_list = []
    
    for i in range(num_points):
            color_predict = rgb_values[int(predictions[i])]
            r_predict, g_predict, b_predict = color_predict
            color_predict_list.append([r_predict, g_predict, b_predict])
    color_predict_list = np.array(color_predict_list)
    seg_colored_points_predict = np.hstack((whole_scene_data[:, :3], color_predict_list))

    print(f"===> start plotting")
    fout = open( f"figures/{args.id}_predict-whole-seg-color.obj", 'w')
    for k in range(num_points):
        fout.write('v %f %f %f %d %d %d\n' % (seg_colored_points_predict[k][0],seg_colored_points_predict[k][1],seg_colored_points_predict[k][2],seg_colored_points_predict[k][3],seg_colored_points_predict[k][4],seg_colored_points_predict[k][5]))

    #################  CREATE TXT FILES WITH XYZ RGB #################


    # np.savetxt(f"figures/{args.id}-target.txt", target)
    # np.savetxt(f"figures/{args.id}-predict.txt", predict)

    # # start plot
    # print(f"===> start plotting")
    #plot_xyz(points, target, name=f"figures/{args.id}-gt.pdf")
    #plot_xyz(points, predict, name=f"figures/{args.id}-predict.pdf")



def plot_xyz(xyz, target, name="figures/figure.pdf"):
    fig = pyplot.figure(figsize = (10,10))
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]
    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)
    for i in range(0,720000):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker="o", s=30, alpha=0.7)
    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    fig.savefig(name, bbox_inches='tight', pad_inches=-0.3, transparent=True)
    pyplot.close()



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP')
    parser.add_argument('--id', type=int, default='8')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test(args)
    