import os
import pickle
import numpy as np






def generate_GD(all_class, dim, root):

    GD_dict = {}

    feat_dict = {x:[] for x in range(all_class)}

    for i in range(len(os.listdir(root))):
        file_path = os.path.join(root,'{}.pkl'.format(i))
        with open(file_path,'rb') as f:
            data = pickle.load(f)

        feat = data['feats'][0]
        label = data['label'][0]
        feat_dict[label].append(feat)
    

    for key in feat_dict:
        feat_list = np.array(feat_dict[key])

        GD_dict[key] = []

        for j in range(dim):
            f_j = feat_list[:,j]
            mean = np.mean(f_j)
            std = np.std(f_j,ddof=1)
            GD_dict[key].append({'mean':mean,'std':std})

    return GD_dict





if __name__=='__main__':
    ALL_CLASS = 100 # total class number of a certain dataset, e.g. 100 for miniImageNet and CIFAR100, 200 for CUB-200, 397 for SUN397
    DIM = 512 # For vit-b/16
    FEATURE_ROOT = '' # please enter your own generated feature folder
    save_name = 'xxxx.pkl' # can replace the xxxx to specific name

    GD_dict = generate_GD(all_class=ALL_CLASS, dim=DIM, root=FEATURE_ROOT)

    with open(save_name,'wb') as f:
        pickle.dump(GD_dict,f,-1)
