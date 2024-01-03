import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import build_transform

from datasets.cifar100 import CIFAR100, CLASSNAME_CFIAR100
from datasets.cub200 import CUB200, CLASSNAME_CUB200
from datasets.miniImageNet import MiniImageNet, CLASSNAME_miniImageNet
from datasets.sun397 import SUN397, CLASSNAME_SUN397
from datasets.cub200_wo_base import CUB200_wo_Base

from models.model import load_clip_to_cpu, CustomCLIP

from tqdm import tqdm







@TRAINER_REGISTRY.register()
class LP_DiF(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]



    def build_data_loader(self):

        self.tfm_train = build_transform(self.cfg, is_train=True)
        self.tfm_test = build_transform(self.cfg, is_train=False)
        
        
        
        self.batch_size_train = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.batch_size_test = self.cfg.DATALOADER.TEST.BATCH_SIZE
        self.num_workers = self.cfg.DATALOADER.NUM_WORKERS

        self.task_id = self.cfg.TRAINER.TASK_ID
 
        self.dataset_name = self.cfg.DATASET.NAME
        self.data_root = self.cfg.DATASET.ROOT
        self.num_classes = self.cfg.DATASET.NUM_CLASSES
        self.num_classes_base = self.cfg.DATASET.NUM_CLASSES_BASE
        self.class_per_task = self.cfg.DATASET.CLASS_PER_TASK
        self.shot = self.cfg.DATASET.NUM_SHOTS
        self.B = self.cfg.DATASET.B
        self.GD_path = self.cfg.DATASET.GD_PATH

        self.task_num = int((self.num_classes-self.num_classes_base)/self.class_per_task)

        if self.num_classes_base>0:
            self.encounter_class_id = self.num_classes_base + self.class_per_task * self.task_id
        else:
            self.encounter_class_id = self.class_per_task * (self.task_id + 1)

        if self.dataset_name=='CIFAR100':

            train_set_task0 = CIFAR100(shot=self.shot,
                                       tfm=self.tfm_train,
                                       task_id = self.task_id,
                                       mode='train',
                                       class_per_task=self.class_per_task,
                                       B=self.B,
                                       GD_path = self.GD_path)
            
            test_set_task0 = CIFAR100(
                                      tfm=self.tfm_test,
                                      task_id = self.task_id,
                                      mode='test',
                                      class_per_task=self.class_per_task, 
                                      )
            

            self.classnames = CLASSNAME_CFIAR100

        elif self.dataset_name=='CUB200':


            train_set_task0 = CUB200(data_root=self.data_root,
                                     shot=self.shot,
                                       tfm=self.tfm_train,
                                       task_id = self.task_id,
                                       mode='train',
                                       class_per_task=self.class_per_task,
                                       B=self.B,
                                       GD_path = self.GD_path)
            
            test_set_task0 = CUB200(data_root=self.data_root,
                                      tfm=self.tfm_test,
                                      task_id = self.task_id,
                                      mode='test',
                                      class_per_task=self.class_per_task)

            
            self.classnames = CLASSNAME_CUB200

        elif self.dataset_name=='miniImageNet':
            train_set_task0 = MiniImageNet(data_root=self.data_root,
                                       tfm=self.tfm_train,
                                       task_id = self.task_id,
                                       mode='train',
                                       class_per_task=self.class_per_task,
                                       B=self.B,
                                       GD_path = self.GD_path)
            
            test_set_task0 = MiniImageNet(data_root=self.data_root,
                                      tfm=self.tfm_test,
                                      task_id = self.task_id,
                                      mode='test',
                                      class_per_task=self.class_per_task
                                      )

            self.classnames = CLASSNAME_miniImageNet

        elif self.dataset_name=='SUN397':

            train_set_task0 = SUN397(data_root=self.data_root,
                                     shot=self.shot,
                                       tfm=self.tfm_train,
                                       task_id = self.task_id,
                                       mode='train',
                                       class_per_task=self.class_per_task,
                                       B=self.B,
                                       GD_path = self.GD_path)
            
            test_set_task0 = SUN397(data_root=self.data_root,
                                      tfm=self.tfm_test,
                                      task_id = self.task_id,
                                      mode='test',
                                      class_per_task=self.class_per_task, 
                                     )

            self.classnames = CLASSNAME_SUN397
        elif self.dataset_name=='CUB200_wo_Base':

            train_set_task0 = CUB200_wo_Base(data_root=self.data_root,
                                             shot=self.shot,
                                            tfm=self.tfm_train,
                                            task_id = self.task_id,
                                            mode='train',
                                            class_per_task=self.class_per_task,
                                            B=self.B,
                                            GD_path = self.GD_path)
            
            test_set_task0 = CUB200_wo_Base(data_root=self.data_root,
                                        tfm=self.tfm_test,
                                        task_id = self.task_id,
                                        mode='test',
                                        class_per_task=self.class_per_task, 
                                        )


            self.classnames = CLASSNAME_CUB200
        
        self.classnames_encountered = self.classnames[:self.encounter_class_id]



        train_loader = torch.utils.data.DataLoader(train_set_task0,batch_size=self.batch_size_train, num_workers=self.num_workers, drop_last=False, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set_task0,batch_size=self.batch_size_test, num_workers=self.num_workers, drop_last=False)

  

        self.train_loader_x = train_loader
        self.val_loader = test_loader  
        self.test_loader = test_loader

        self.lab2cname = {x:self.classnames_encountered[x] for x in range(len(self.classnames_encountered))}  


    def build_model(self):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        self.clip_model = clip_model
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames_encountered, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model,device_ids=[0, 1, 2,3,4,5,6])

        
        self.lambda_o = self.cfg.TRAINER.LAMBDA_O

    def forward_backward(self, batch):

        if self.task_id==0:
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)

        else:
            image, label,pseudo_feat,pseudo_label = batch

            pseudo_feat = pseudo_feat.view(-1,512)
            pseudo_label = pseudo_label.view(-1)

            image = image.to(self.device)
            label = label.to(self.device)

            pseudo_label = pseudo_label.to(self.device)
            pseudo_feat = pseudo_feat.to(self.device)

            
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output,output_pseudo = self.model(image,pseudo_feat)

                loss = F.cross_entropy(torch.cat((output,output_pseudo)), torch.cat((label,pseudo_label)),reduction='none')


                weight_n = torch.ones((image.shape[0]))
                weight_o = torch.ones((pseudo_feat.shape[0])) * self.lambda_o
                weight = torch.cat((weight_n,weight_o)).half()

                

                loss = loss * (weight.to(loss.device).detach())
                loss = loss.mean()
                self.model_backward_and_update(loss)


        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test" 
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            output = self.model_inference(image)
            self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)











