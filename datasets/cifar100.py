import random
import numpy as np
import pickle
import os
import torchvision
from torch.utils.data import Dataset as TorchDataset



CLASSNAME_CFIAR100 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


class CIFAR100(TorchDataset):

    def __init__(self, tfm, task_id,mode,shot=5, class_per_task=5,B=8, GD_path = '') -> None:


        self.class_per_task = class_per_task
        self.novel_task_len = int(40/class_per_task)
        self.task_len = self.novel_task_len+1
        self.mode = mode
        self.task_id = task_id
        self.B = B

        if mode=='train':

            with open(GD_path,'rb') as f:
                self.GD = pickle.load(f)
            task_split = [[] for x in range(9)]

            for i in range(60):
                task_split[0].append(i)

            for i in range(1,9):
                for j in range(5):
                    task_split[i].append(60+(i-1)*5+j)

            select_class_id = task_split[task_id]
       
            cifar100 = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in select_class_id}
            self.end_class_id = select_class_id[-1]


            for i in range(len(cifar100)):
                image,label = cifar100[i]

                if label in self.class_idx_dict:

                    self.class_idx_dict[label].append(i)

            self.data = []
    
            for c in select_class_id:
                idx_list = self.class_idx_dict[c]
                if c>=60:
                    idx_list = random.sample(idx_list,shot)
                for id in idx_list:
                    self.data.append(cifar100[id])

            self.shot = shot

            self.len = len(self.data)
        else:

            self.data = []
            task_to_id_end = {0:60}
            start = 65
            for i in range(1,9):
                task_to_id_end[i]=start
                start+=5

            select_class_id=[x for x in range(task_to_id_end[task_id])]

            self.end_class_id = select_class_id[-1]

            cifar100 = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=False,transform=tfm)

            self.class_idx_dict = {x:[] for x in select_class_id}
            for i in range(len(cifar100)):
                image,label = cifar100[i]

                if label in select_class_id:

                    self.data.append(cifar100[i])
            self.len = len(self.data)



    def __getitem__(self,index):
        if self.mode=='train' and self.task_id>0:
            pseudo_label = []
            pseudo_feat_list = []
            have_select = 0
            while True:
           
                select_class = random.randint(0,self.end_class_id-self.class_per_task)

                GD_all = self.GD[select_class]
                pseudo_feat = []
                for dim in range(len(GD_all)):
                    dim_param = GD_all[dim]

                    mean = dim_param['mean']

                    std = dim_param['std']
                    pseudo_value = np.random.normal(mean, std, 1)[0]
                    pseudo_feat.append(pseudo_value)
                pseudo_feat = np.array(pseudo_feat)

                pseudo_feat = pseudo_feat / np.linalg.norm(pseudo_feat)

                pseudo_feat_list.append(pseudo_feat)
                pseudo_label.append(select_class)
      
                have_select+=1
                if have_select==self.B:
                    break
            pseudo_label = np.array(pseudo_label)
            pseudo_feat_list = np.array(pseudo_feat_list)
            
            return self.data[index][0],self.data[index][1],pseudo_feat_list,pseudo_label
        return self.data[index]


    def __len__(self):
        return self.len