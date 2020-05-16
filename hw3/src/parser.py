from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../hw3_data/',
                    help="root path to data directory")
    parser.add_argument('--model_type', type=str, default='gan',
                        choices=['gan', 'acgan', 'dann'],
                        help='three model types: gan, acgan or dann')
    parser.add_argument('--image_size', default=64, type=int)

    # model parameters
    parser.add_argument('--nz', default=100, type=int,
                    help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument('--ngf', default=64, type=int,
                    help="Size of feature maps in generator")
    parser.add_argument('--ndf', default=64, type=int,
                    help="Size of feature maps in discriminator")
    parser.add_argument('--nc', default=3, type=int,
                    help="Number of channels in the training images. For color images this is 3")

    # save folder
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--save_samples_dir', type=str, default='samples')

    # training parameters for gan
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--random_seed', default=999, type=int,
                    help='Set random seed for reproducibility')
    parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--lr_D', default=0.0004, type=float,
                    help="initial learning rate for discriminator")
    parser.add_argument('--lr_G', default=0.0001, type=float,
                    help="initial learning rate for discriminator")
    parser.add_argument('--beta1', default=0.5, type=float,
                    help="initial learning rate for discriminator")
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")

    # training parameters for ac gan
    parser.add_argument("--n_classes", type=int, default=2,
                    help="number of classes for dataset, smiling or not")

    # others
    parser.add_argument("--samples_num", type=int, default=32,
                    help="how many sample images wanted to generate")

    # DANN parameters
    parser.add_argument("--digit_classes", type=int, default=10,
                    help="number of classes for digit, 0-9")
    parser.add_argument("--num_domain", type=int, default=2,
                    help="number of classes for digit, 0-9")
    parser.add_argument('--source_dataset', type=str, default='mnistm')
    parser.add_argument('--target_dataset', type=str, default='svhn')
    parser.add_argument('--dann_type', type=str, default='both',
                        choices=['both', 'single'],
                        help='dann training: only source, only target, or both')
    parser.add_argument('--lr', default=0.001, type=float,
                    help="initial learning rate for dann")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations for dann")

    # resume trained model
    parser.add_argument('--resume_folder', type=str, default='../dann/latest_',
                    help="path to the trained model")


    # ADDA parameters
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--src_model', default='../adda/5/', help='A model in trained_models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k-disc', type=int, default=1)
    parser.add_argument('--k-clf', type=int, default=10)
    parser.add_argument('--lr_tar', default=0.00001, type=float)
    parser.add_argument('--lr_adda_d', default=0.00001, type=float)


    # hw3_p3 parameters
    parser.add_argument('--target_domain', type=str, default='mnistm')
    parser.add_argument('--save_folder', type=str, default='.')


    args = parser.parse_args()

    return args
