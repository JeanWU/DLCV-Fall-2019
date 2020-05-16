from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='hw2_data/',
                    help="root path to data directory")
    parser.add_argument('--img_dir', type=str, default='hw2_data/train/img',
                    help="root path to image directory")
    parser.add_argument('--seg_dir', type=str, default='hw2_data/train/seg',
                    help="root path to segmentation mask directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int,
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # resume trained model
    parser.add_argument('--resume', type=str, default='log/model_best.pth.tar',
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--save_improved_dir', type=str, default='log_improved')
    parser.add_argument('--save_predict', type=str, default='')

    args = parser.parse_args()

    return args
