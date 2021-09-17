import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from models import ResNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils


cudnn.enabled = True
cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parse Arguments
parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--datadir', metavar='DATA', default='/data/home/yanghao/dataset/kitti_dataset',
                    choices=root_dir,
                    help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                    help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                    help='number of sparse depth samples (default: 0)')
parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                    help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                    help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')
parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2', choices=decoder_names,
                    help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run (default: 15)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                    help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                    help='evaluate model on validation set')
parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                    help='not to use ImageNet pre-trained weights')
parser.set_defaults(pretrained=True)

args = parser.parse_args()

print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()





global args, best_result, output_directory, train_csv, test_csv


start_epoch = 0


# train mode 
# create new model
train_loader, val_loader = create_data_loaders(args)
print("loading train data")


print('base_dir is',self.base_dir)
print('depth_path is',os.path.join(self.base_dir, 'data_depth_annotated', mode))
print('lidar_path is',os.path.join(self.base_dir, 'data_depth_velodyne', mode))
print('depth_path is',os.path.join(self.base_dir, 'val_selection_cropped', 'groundtruth_depth'))
print('lidar_path is',os.path.join(self.base_dir, 'val_selection_cropped', 'velodyne_raw'))
print('image_path is',os.path.join(self.base_dir, 'val_selection_cropped', 'image'))

datadir

train_dir = os.path.join(args.datadir, 'data_depth_annotated', mode))


