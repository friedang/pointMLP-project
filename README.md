This is an unofficial project to improve and test PointMLP's performance for scene segmentation. Originally, PointMLP was developed for 3d object classification and segmentation on the ShapeNet dataset - outperforming heavyweight transformer models. 

We - Friedrich Dang, Begüm Altunbas, Görkem Güzeler and Han Keçeli tested several modifications of PointMLP on Stanford's 3D Indoor Scene Dataset (S3DIS). We improved its performance by 3-5% across various categories by exchanging the non-learnable max-pooling aggregation function with self-attention layers for the global feature extraction. Together with an MLP, feature density is kept high and a former upsampling procedure was made redundant.

Some test results on S3DIS for some of our modifications are summarized below.
| Modification | Accuracy | Instance mIOU | Class mIOU |
|----------|----------|----------|----------|
| None   | 0.67   | 0.65   | 0.52 |
| Random Sampling   | 0.68   | 0.67   | 0.53 |
| Poisson + FPS Sampling   | 0.68   | 0.64   | 0.51 |
| GAM Non-Linear Normalization   | 0.61   | 0.59   | 0.47 |
| Self-Attention global context   | 0.70   | 0.69   | 0.56 |
| Multihead Self-Attention - color and global context  | 0.70   | 0.69   | 0.57 |


## Install

```bash
conda create -n pointmlp python=3.7 -y
conda activate pointmlp
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
# if you are using Ampere GPUs (e.g., A100 and 30X0), please install compatible Pytorch and CUDA versions, like:
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```


## Useage

### Scene segmentation S3DIS

- Make data folder under part_segmentation/data
- Download our subsample of S3DIS dataset from https://drive.google.com/drive/folders/1CW5qicxlE5tkpRiczTI84qDZxQojymD0?usp=share_link
- Download the S3DIS dataset to that folder and save it as custom_s3 (arrange root path in main.py accordingly).
- To perform evaluation please download model checkpoints from the same drive link and place it under /checkpoints folder (arrange paths accoringly).
  
- Train & Test pointMLP
```bash
# train pointMLP
cd part_segmentation
python main.py --model pointMLP
# please add other paramemters as you wish.
# eval pointMLP
python main.py --model pointMLP --eval True
```
- Visualize pointMLP on S3DIS dataset
```bash
# visualize pointMLP
cd part_segmentation
python visualize_test.py --log_dir pointmlp --test_area 5 --visual
# make sure that you have log/sem_seg/pointmlp/visual & log/sem_seg/pointmlp/logs folders created and correct checkpoint in checkpoints folder.
```
visualize_test.py saves .obj files inside log folder for 11 test instances with gt and predictions. To visualize these .obj files we have another script open3d_show_obj.py (arrange the .obj file path accordingly).
Beware that Meshlab shows all points in white color thus it is recomended to use open3d_show_obj.py. Example .obj files are placed in the google drive link.

P.S: The data preprocessing (creating custom_s3/x.npy files), visualization and dataloader codes are based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch and modified according to our project needs.


### Part segmentation ShapeNet

- Make data folder and download the dataset
```bash
cd part_segmentation
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```
- Train pointMLP
```bash
# train pointMLP
python main-original.py --model pointMLP
# please add other paramemters as you wish.
```

### Classification ModelNet40
**Train**: The dataset will be automatically downloaded, run following command to train.

By default, it will create a folder named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
cd classification_ModelNet40
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```


To conduct voting testing, run
```bash
# please modify the msg accrodingly
python voting.py --model pointMLP --msg demo
```


### Classification ScanObjectNN

The dataset will be automatically downloaded

- Train pointMLP/pointMLPElite 
```bash
cd classification_ScanObjectNN
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.


## Acknowledgment

Our implementation is mainly based on the PointMLP codebase. We gratefully thank the authors for their wonderful works.

[S3DIS](http://buildingparser.stanford.edu/dataset.html)
```bash
@ARTICLE{2017arXiv170201105A,
   author = {{Armeni}, I. and {Sax}, A. and {Zamir}, A.~R. and {Savarese}, S.
	},
    title = "{Joint 2D-3D-Semantic Data for Indoor Scene Understanding}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1702.01105},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Robotics},
     year = 2017,
    month = feb,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170201105A},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
[PointMLP](https://github.com/ma-xu/pointMLP-pytorch)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-network-design-and-local-geometry-1/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=rethinking-network-design-and-local-geometry-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-network-design-and-local-geometry-1/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=rethinking-network-design-and-local-geometry-1)


[![github](https://img.shields.io/github/stars/ma-xu/pointMLP-pytorch?style=social)](https://github.com/ma-xu/pointMLP-pytorch)


<div align="left">
    <a><img src="images/smile.png"  height="70px" ></a>
    <a><img src="images/neu.png"  height="70px" ></a>
    <a><img src="images/columbia.png"  height="70px" ></a>
</div>

 [open review](https://openreview.net/forum?id=3Pbra-_u76D) | [arXiv](https://arxiv.org/abs/2202.07123) | Primary contact: [Xu Ma](mailto:ma.xu1@northeastern.edu)

<div align="center">
  <img src="images/overview.png" width="650px" height="300px">
</div>

## LICENSE
PointMLP is under the Apache-2.0 license. 






