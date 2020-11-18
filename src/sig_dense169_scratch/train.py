# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import os.path as osp
from typing import Optional
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import attr
import numpy as np
from .config import TrainerConfig, ClusterConfig
#from .transforms import get_transforms
#from .samplers import RASampler
#import torchvision.models as models
from .model import MURA_Net
#import timm
#import tqdm
from sklearn.metrics import cohen_kappa_score
import csv
from .dataset import MURA_Dataset
import cv2
import torch.nn.functional as F
import pandas as pd

def conv_numpy_tensor(output):
    """Convert CUDA Tensor to numpy element"""
    return output.data.cpu().numpy()

@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    accuracy:float
    model: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["lr_scheduler"] = self.lr_scheduler.state_dict()
        data["accuracy"] = self.accuracy
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState") -> "TrainerState":
        data = torch.load(filename)
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(data["optimizer"])
        data["optimizer"] = optimizer

        lr_scheduler = default.lr_scheduler
        lr_scheduler.load_state_dict(data["lr_scheduler"])
        data["lr_scheduler"] = lr_scheduler
        return cls(**data)


class Trainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig) -> None:
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg

    def __call__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._init_state()
        final_acc = self._train()
        return final_acc

    def __eval__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        #self._init_state()
        self._init_state_test()
        final_acc = self._test()
        return final_acc

    def __show__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._show()


    def checkpoint(self, rm_init=True):
        save_dir = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id))
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(osp.join(save_dir, "checkpoint.pth"))
        self._state.save(osp.join(save_dir, "checkpoint_"+str(self._state.epoch)+".pth"))
        if rm_init:
            os.remove(self._cluster_cfg.dist_url[7:])  
        empty_trainer = Trainer(self._train_cfg, self._cluster_cfg)
        return empty_trainer

    def _setup_process_group(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_state_test(self) -> None:
        """
            Initialize the state and load it from an existing checkpoint if any
            """
        torch.manual_seed(0)
        np.random.seed(0)

        Input_size_Image = self._train_cfg.input_size

        # Test_size=Input_size_Image
        print("Input size : " + str(Input_size_Image))
        print("Test size : " + str(Input_size_Image))
        print("Initial LR :" + str(self._train_cfg.lr))
        print("Create data loaders", flush=True)

        # test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths
        #                         , input_size=self._train_cfg.input_size, part=self._train_cfg.mura_part, train=False,
        #                         test=False)

        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.test_image_paths
                                , input_size=self._train_cfg.input_size, part=self._train_cfg.mura_part, train=False,
                                test=False)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False,
            num_workers=(self._train_cfg.workers - 1),  # sampler=test_sampler, Attention je le met pas pour l instant
        )
        model = MURA_Net()
        #model = timm.create_model('efficientnet_b7', pretrained=False)
        #model = models.resnet152(pretrained=False)
        #model = models.densenet161(pretrained=True)
        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, 2)
        #num_ftrs = model.classifier.in_features
        #model.classifier = nn.Linear(num_ftrs, 2)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 75], gamma=0.25)
        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000)

        self._state = TrainerState(
            epoch=0, accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint_{0}.pth".format(str(self._train_cfg.load_epoch)))
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)
            print("model_load")

    def _init_state(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        torch.manual_seed(0)
        np.random.seed(0)
        Input_size_Image=self._train_cfg.input_size
        
        #Test_size=Input_size_Image
        print("Input size : "+str(Input_size_Image))
        print("Test size : "+str(Input_size_Image))
        print("Initial LR :"+str(self._train_cfg.lr))

        print("Create data loaders", flush=True)
        train_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.train_image_paths
                                 ,input_size = self._train_cfg.input_size , part=self._train_cfg.mura_part, train=True, test=False)

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers-1),
            shuffle=True
            #sampler=train_sampler,
        )

        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths
                                , input_size = self._train_cfg.input_size, part=self._train_cfg.mura_part, train=False, test=False )

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),#sampler=test_sampler, Attention je le met pas pour l instant
        )

        #model = timm.create_model('efficientnet_b7', pretrained=False)
        #model = models.resnet152(pretrained=False)
        #model = models.densenet161(pretrained=False)
        model = MURA_Net()
        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, 2)
        #num_ftrs = model.classifier.in_features
        #model.classifier = nn.Linear(num_ftrs, 2)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(self.device)

        #optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

        optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=1e-5 )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 75], gamma=0.5)

        self._state = TrainerState(
            epoch=0,accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)
            print("model_load")

    def _train(self) -> Optional[float]:
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.binary_cross_entropy()
        #loss = F.binary_cross_entropy(outputs, labels, weight=weights)
       # running_loss += loss

        print_freq = 20
        acc = None
        max_accuracy=0.0
        # Start from the loaded epoch
        start_epoch = self._state.epoch
        previous_loss = 1e10
        lr = self._train_cfg.lr
        for epoch in range(start_epoch, self._train_cfg.epochs):
            print(f"Start epoch {epoch}", flush=True)
            self._state.model.train()
            #self._state.lr_scheduler.step()
            self._state.epoch = epoch
            running_loss = 0.0
            current_loss = 0.0
            count=0
            total_step = len(self._train_loader)
            for i, data in enumerate(self._train_loader):
                inputs, labels, _, body_part = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self._state.model(inputs)
                #loss = criterion(outputs, labels)
                outputs = outputs.reshape(-1)
                loss = F.binary_cross_entropy(outputs, labels.float())

                self._state.optimizer.zero_grad()
                loss.backward()
                self._state.optimizer.step()

                running_loss += loss.item()
                count=count+1
                current_loss += loss.item()

                if i % print_freq == print_freq - 1:
                    print('Epoch [{0}/{1}], Step [{2}/{3}], Loss: {4:.4f}'
                          .format(epoch + 1, self._train_cfg.epochs, i + 1, total_step, running_loss/print_freq))
                    running_loss = 0.0

            # update learning rate
            if current_loss/count > previous_loss:
                # if val_loss > previous_loss:
                lr= lr * 0.5
                #lr = lr * opt.lr_decay
                # 第二种降低学习率的方法:不会有moment等信息的丢失
                for param_group in self._state.optimizer.param_groups:
                    param_group['lr'] = lr#self._train_cfg.lr
                    print("loss decay {0} , decay factor{1}".format(lr, 0.5 ))

            # previous_loss = val_loss
            previous_loss = current_loss/count
            current_loss=0
            count=0

            if epoch%1 == 0 or epoch == 0:
                print("Start evaluation of the model", flush=True)
                correct = 0
                total = 0
                count=0.0
                running_val_loss = 0.0
                self._state.model.eval()
                with torch.no_grad():
                    for data in self._test_loader:
                        images, labels, _, body_part = data

                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self._state.model(images)
                        outputs = outputs.reshape(-1)
                        loss_val = F.binary_cross_entropy(outputs, labels.float())
                        #loss_val = criterion(outputs, labels)
                        #_, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)

                        predicted = (outputs > 0.5).type(torch.LongTensor).reshape(-1)
                        predicted = predicted.to(self.device)
                        correct += (predicted == labels).sum().item()
                        running_val_loss += loss_val.item()
                        count=count+1.0

                acc = correct / total
                ls_nm=running_val_loss/count

                print('Epoch [{0}/{1}], Step [{2}/{3}], Val Loss: {4:.4f} Val Acc: {5:.2f}'
                      .format(epoch + 1, self._train_cfg.epochs, i + 1, total_step, ls_nm, acc))

                self._state.accuracy = acc
                max_accuracy=np.max((max_accuracy, acc))

                # Save for best accuracy models
                if acc >= max_accuracy :
                    print("Epoch [{0}/{1}], Save Best Model[accuracy {0}]".format(epoch + 1, self._train_cfg.epochs, acc))
                    self.checkpoint(rm_init=False)

                if epoch==self._train_cfg.epochs-1:
                    return acc

    def _test(self) -> Optional[float]:
        self._state.model.eval()
        print("Start evaluation of the model", flush=True)

        correct = 0
        total = 0
        count = 0.0
        results = []
        mura_result = list()
        with torch.no_grad():
            for data in self._test_loader:
                mura_rows=dict()
                images, labels, img_path, body_part = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self._state.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                count = count + 1.0

                batch_results = [(labels_, predicted_, body_part_, img_path_[2:]) for labels_, predicted_, body_part_, img_path_ in zip(labels.cpu().numpy(), predicted.cpu().numpy(), body_part, img_path)]
                results += batch_results

        df = pd.DataFrame(results)
        df_r = df[[3,1]]
        df_r.to_csv(self._train_cfg.output, header=False, index=False)
        acc = correct / total
        np_result = np.array(results)
        kappa_score = cohen_kappa_score(np_result[:,0], np_result[:,1])

        print("All cohen kappa", kappa_score)
        if self._train_cfg.mura_part == 'all':
            XR_type_list = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
        else:
            XR_type_list = [self._train_cfg.mura_part]
        for xr_type in XR_type_list:
            xr_type_correct = 0
            xr_type_result = np_result[np.where(np_result[:, 2] == xr_type)][:,0:2]
            xr_type_correct += (xr_type_result[:, 0] == xr_type_result[:, 1]).sum().item()
            xr_type_cohen_kappa = cohen_kappa_score(xr_type_result[:, 0], xr_type_result[:, 1])
            print('cohen_kappa {0} : {1:.2f}'.format(xr_type,xr_type_cohen_kappa ))
            print('ACC {0} : {1:.2f}'.format(xr_type,xr_type_correct/xr_type_result.shape[0] ))

        print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
        self._state.accuracy = acc
        return acc

    def _show(self) -> Optional[float]:

        print("Create data loaders", flush=True)
        train_set = MURA_Dataset(self._train_cfg.data_root,
                                 self._train_cfg.data_root + self._train_cfg.train_image_paths
                                 , input_size=self._train_cfg.input_size, part=self._train_cfg.mura_part, train=True,
                                 test=False)

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers - 1),
            shuffle=True
            # sampler=train_sampler,
        )

        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths
                                , input_size=self._train_cfg.input_size, part=self._train_cfg.mura_part, train=False,
                                test=False)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False,
            num_workers=(self._train_cfg.workers - 1),  # sampler=test_sampler, Attention je le met pas pour l instant
        )

        for i, data in enumerate(self._train_loader):
            inputs, labels, file_path, body_part = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            for i in range(inputs.shape[0]):
                img_data = inputs.cpu().numpy()[i]
                img_data = np.transpose(img_data, (1,2,0))
                #img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                cv2.imshow("_".join([file_path[i].split('/')[4],file_path[i].split('/')[5],file_path[i].split('/')[6]]), img_data)
                print("_".join([file_path[i].split('/')[4],file_path[i].split('/')[5],file_path[i].split('/')[6]]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
