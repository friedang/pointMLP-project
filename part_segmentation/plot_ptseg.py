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
from util.S3DISDataLoader import S3DISDataset
#import pyntcloud
import pandas as pd
#import open3d as o3d

# import matplotlib.colors as mcolors
# def_colors = mcolors.CSS4_COLORS
# colrs_list = []
# np.random.seed(2021)
# for k, v in def_colors.items():
#     colrs_list.append(k)
# np.random.shuffle(colrs_list)
colrs_list = [
    "C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet"
]


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
    #test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    
    root = '/cluster/51/pointMLP_data/data/custom_s3'
    
    test_data= S3DISDataset(split='test', data_root=root, num_point=800)
    
    print("===> The number of test data is:%d", len(test_data))
    # Try to load models
    print("===> Create model...")
    num_part = 13
    device = torch.device("cuda" if args.cuda else "cpu")
    model = models.__dict__[args.model](num_part).to(device)
    print("===> Load checkpoint...")
    from collections import OrderedDict
    # state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
    #                         map_location=torch.device('cpu'))['model']
    state_dict = torch.load("checkpoints/pointMLP_self_attention/best_insiou_model.pth",
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)
    print("===> Start evaluate...")
    model.eval()
    num_classes = 13
    points, target, norm_plt, color = test_data.__getitem__(args.id)
    print( "POINT SIZE",   points.shape)

    points = torch.tensor(points).unsqueeze(dim=0)
    #label = torch.tensor(label).unsqueeze(dim=0)
    target = torch.tensor(target).unsqueeze(dim=0)
    norm_plt = torch.tensor(norm_plt).unsqueeze(dim=0)
    color = torch.tensor(color).unsqueeze(dim=0)

    points, target, norm_plt,color  = Variable(points.float()), Variable(target.long()), Variable(norm_plt.float()),Variable(color.float())
    points = points.transpose(2, 1)
    norm_plt = norm_plt.transpose(2, 1)
    color = color.transpose(2, 1)
    points, target, norm_plt, color = points.cuda(non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True),color.cuda(non_blocking=True)
    with torch.no_grad():
            #cls_lable = to_categorical(label, num_classes)
            predict = model(points, norm_plt,color)  # b,n,50
    # up to now, points [1, 3, 2048]  predict [1, 2048, 50] target [1, 2048]
    predict = predict.max(dim=-1)[1]
    predict = predict.squeeze(dim=0).cpu().data.numpy()  # 2048
    target = target.squeeze(dim=0).cpu().data.numpy()   # 2048
    points = points.transpose(2, 1).squeeze(dim=0).cpu().data.numpy() #[2048,3]


    np.savetxt(f"figures/{args.id}-point.txt", points)
    
    #################  CREATE TXT FILES WITH XYZ RGB #################
    color_target_list = []
    color_predict_list = []
    for i in range(len(target)):
        color_target = rgb_values[target[i]]
        color_predict = rgb_values[predict[i]]
        
        r_predict, g_predict, b_predict = color_predict
        r_target, g_target, b_target = color_target
        color_predict_list.append([r_predict, g_predict, b_predict] )
        color_target_list.append([r_target, g_target, b_target] )

    color_target_list = np.array(color_target_list)
    color_predict_list = np.array(color_predict_list)
    points_pred = np.hstack((points, color_predict_list))
    points_target = np.hstack((points, color_target_list))
    print( "SHAPEEE NEW OF POINTSSSS: ",points_pred.shape )
    print( "SHAPEEE NEW OF POINTSSSS: ",points_target.shape)

    np.savetxt(f"figures/{args.id}-predict-color.txt", points_pred)
    np.savetxt(f"figures/{args.id}-target-color.txt", points_target)
    
    #################  CREATE TXT FILES WITH XYZ RGB #################
    np.savetxt(f"figures/{args.id}-target.txt", target)
    np.savetxt(f"figures/{args.id}-predict.txt", predict)

    # start plot
    print(f"===> start plotting")
    plot_xyz(points, target, name=f"figures/{args.id}-gt.pdf")
    plot_xyz(points, predict, name=f"figures/{args.id}-predict.pdf")



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
    for i in range(0,800):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker="o", s=30, alpha=0.7)
    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    fig.savefig(name, bbox_inches='tight', pad_inches=-0.3, transparent=True)
    pyplot.close()

# def plot_xyz_pcd(xyz, target, name="figures/figure.pcd"):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.colors = o3d.utility.Vector3dVector(np.array([colrs_list[int(label)] for label in target]))
#     o3d.io.write_point_cloud(name, pcd)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP')
    parser.add_argument('--id', type=int, default='1')
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
    