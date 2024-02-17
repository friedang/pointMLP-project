from open3d import *
import numpy as np
import open3d as o3d

# Load OBJ file
obj_file_path = "/Users/begumaltunbas/Desktop/visualize_ml43d/Area_5_hallway_15_pred.obj"
#obj_file_path = "/Users/begumaltunbas/Desktop/visualize_ml43d/Area_5_hallway_15_gt_raw.obj"

# Read all vertices and colors from the OBJ file
with open(obj_file_path, 'r') as obj_file:
    lines = obj_file.readlines()
    vertices = [list(map(float, line.split()[1:4])) for line in lines if line.startswith('v')]
    colors = [list(map(int, line.split()[4:7])) for line in lines if line.startswith('v')]

# Create a PointCloud object with colors
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)
point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors)/255)  # Normalize RGB to [0, 1]

# Visualize the entire point cloud
o3d.visualization.draw_geometries([point_cloud])