from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='hw4')

    # Datasets parameters
    parser.add_argument('--csv_path', type=str, default='../hw4_data/TrimmedVideos/label/',
                    help="direct path to csv file")
    parser.add_argument('--video_path', type=str, default='../hw4_data/TrimmedVideos/video/',
                    help="root path to image directory")
    parser.add_argument('--full_video_label_path', type=str, default='../hw4_data/FullLengthVideos/labels/',
                    help="direct path to csv file")
    parser.add_argument('--full_video_path', type=str, default='../hw4_data/FullLengthVideos/videos/',
                    help="root path to image directory")
    parser.add_argument('--rescale_factor', default=1, type=float,
                    help="rescale video frame to save computation resources")
    parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=64, type=int,
                    help="test batch size")
    parser.add_argument('--data_batch', default=1, type=int,
                    help="batch size for data loader")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--image_size', default=229, type=int)

    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--output_csv', type=str, default='output')

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")

    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    # parameters for bidirectional LSTM
    parser.add_argument('--hidden_size', default=512, type=int,
                    help="hidden size for bidirectional LSTM")
    parser.add_argument('--num_layers', default=2, type=int,
                    help="number of LSTM layers")

    # parameters for plot feature tsne
    parser.add_argument('--feature_dir', type=str, default='p1_log/train_features',
                    help="")

    # parameters for seq2seq
    parser.add_argument('--load_model_path', type=str, default='p2_log/BiRNN_best.pth.tar',
                    help="direct path to p2 best model")
    parser.add_argument('--extract_batch', default=750, type=int,
                    help="batch size for feature extractor")
    parser.add_argument('--crop_batch', default=300, type=int,
                    help="batch size for feature extractor")








    args = parser.parse_args()

    return args
