from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='image inpainting')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../Data_Challenge2',
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--train_batch', default=1, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=1, type=int,
                    help="test batch size")
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--lr', default=2e-4, type=float,
                    help="initial learning rate")
    parser.add_argument('--lr_finetune', default=5e-05, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_folder', type=str, default='log')

    # Others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--ground_truth', type=str, default='../Data_Challenge2/test_gt')

    # parameters for test
    parser.add_argument('--test_dir', type=str, default='../Data_Challenge2/test',
                    help="root path to testing data directory")



    args = parser.parse_args()

    return args
