# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from sig_dense169_scratch import TrainerConfig, ClusterConfig, Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


def run(input_sizes, learning_rate, epochs, batch, workers, shared_folder_path, job_id, data_root, train_image_paths
        , test_image_paths, mura_part, load_epoch):
    # Distribution config but it is not used
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    data_folder_Path = None
    if Path(str(shared_folder_path)).is_dir() == False:
        raise RuntimeError("No shared folder available")

    train_cfg = TrainerConfig(
        data_folder=str(data_folder_Path),
        epochs=epochs,
        lr=learning_rate,
        input_size=input_sizes,
        batch_per_gpu=batch,
        save_folder=str(shared_folder_path),
        workers=workers,
        job_id=job_id,
        data_root=data_root,
        train_image_paths=train_image_paths,
        test_image_paths=test_image_paths,
        mura_part=mura_part,
        load_epoch=load_epoch
    )

    trainer = Trainer(train_cfg, cluster_cfg)

    # The code should be launch on each GPUs
    try:
        val_accuracy = trainer.__call__()
        print(f"Validation accuracy: {val_accuracy}")
    except Exception as e:
        print("Job failed : {0}, {1}".format(e, e.__traceback__.tb_lineno))


def test(input_sizes, learning_rate, epochs, batch, workers, shared_folder_path, job_id, data_root, train_image_paths,
         test_image_paths, mura_part, load_epoch, prediction):
    # Distribution config but it is not used
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    data_folder_Path = None
    print(os.getcwd())
    if Path(str(shared_folder_path)).is_dir() == False:
        raise RuntimeError("No shared folder available")

    train_cfg = TrainerConfig(
        data_folder=str(data_folder_Path),
        epochs=epochs,
        lr=learning_rate,
        input_size=input_sizes,
        batch_per_gpu=batch,
        save_folder=str(shared_folder_path),
        workers=workers,
        job_id=job_id,
        data_root=data_root,
        train_image_paths=train_image_paths,
        test_image_paths=test_image_paths,
        mura_part=mura_part,
        load_epoch=load_epoch,
        output = prediction
    )

    trainer = Trainer(train_cfg, cluster_cfg)

    # The code should be launch on each GPUs
    try:
        val_accuracy = trainer.__eval__()
        print(f"Validation accuracy: {val_accuracy}")
    except Exception as e:
        print("Job failed : {0}, {1}".format(e, e.__traceback__.tb_lineno))


def show(input_sizes, learning_rate, epochs, batch, workers, shared_folder_path, job_id, data_root,
         train_image_paths,
         test_image_paths, mura_part, load_epoch):
    # Distribution config but it is not used
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    data_folder_Path = None
    if Path(str(shared_folder_path)).is_dir() == False:
        raise RuntimeError("No shared folder available")

    train_cfg = TrainerConfig(
        data_folder=str(data_folder_Path),
        epochs=epochs,
        lr=learning_rate,
        input_size=input_sizes,
        batch_per_gpu=batch,
        save_folder=str(shared_folder_path),
        workers=workers,
        job_id=job_id,
        data_root=data_root,
        train_image_paths=train_image_paths,
        test_image_paths=test_image_paths,
        mura_part=mura_part,
        load_epoch=load_epoch
    )

    trainer = Trainer(train_cfg, cluster_cfg)

    # The code should be launch on each GPUs
    try:
        trainer.__show__()
        # print(f"Validation accuracy: {val_accuracy}")
    except Exception as e:
        print("Job failed : {0}, {1}".format(e, e.__traceback__.tb_lineno))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for ResNet50 FixRes",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('valid_image_paths', type=str, help='valid_image_paths')
    parser.add_argument('predictions', type=str, help='output csv')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='base learning rate')
    parser.add_argument('--input-size', default=320, type=int, help='images input size')
    parser.add_argument('--epochs', default=70, type=int, help='epochs')
    parser.add_argument('--batch', default=10, type=int, help='Batch by GPU')
    parser.add_argument('--gpu_node', default='0', type=str, help='GPU nodes')
    parser.add_argument('--workers', default=1, type=int, help='Numbers of CPUs')
    parser.add_argument('--shared-folder-path', default='./src/shared_folder', type=str, help='Shared Folder')
    parser.add_argument('--job-id', default='sig_dense_no', type=str,
                        help='id of the execution')
    parser.add_argument('--data_root', default='./', type=str, help='id of the execution')
    parser.add_argument('--train_image_paths', default='MURA-v1.1/train_image_paths.csv', type=str,
                        help='id of the execution')
    parser.add_argument('--test_image_paths', default='MURA-v1.1/valid_image_paths.csv', type=str,
                        help='id of the execution')
    parser.add_argument('--mura_part', type=str, default="XR_FINGER",
                        help='type of mura dataset')  # #'ALL XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    parser.add_argument('--load-epoch', type=str, default="58", help='# epoch')

    #MURA-v1.1/valid/XR_ELBOW/patient99999/study1_positive/,1
    #MURA-v1.1/valid/XR_ELBOW/patient99999/study2_negative/,0

    args = parser.parse_args()
    if args.gpu_node == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    elif args.gpu_node == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    elif args.gpu_node == '0,1':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


    print("arg1 : {0}".format(args.valid_image_paths))
    print("arg2 : {0}".format(args.predictions))

    # run(args.input_size
    #     , args.learning_rate
    #     , args.epochs
    #     , args.batch
    #     , args.workers
    #     , args.shared_folder_path
    #     , args.job_id
    #     , args.data_root
    #     , args.train_image_paths
    #     , args.test_image_paths
    #     , args.mura_part
    #     , args.load_epoch)

    test(args.input_size
         , args.learning_rate
         , args.epochs
         , args.batch
         , args.workers
         , args.shared_folder_path
         , args.job_id
         , args.data_root
         , args.train_image_paths
         , args.valid_image_paths
         , args.mura_part
         , args.load_epoch
         , args.predictions)

    # cohen_kappa XR_HAND : 0.31
    # ACC XR_HAND : 0.69

