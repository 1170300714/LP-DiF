import random
import numpy as np
import pickle
import os
from torch.utils.data import Dataset as TorchDataset
from PIL import Image



CLASSNAME_SUN397 = [
    'abbey',
    'airplane cabin',
    'airport terminal',
    'alley',
    'amphitheater',
    'amusement arcade',
    'amusement park',
    'anechoic chamber',
    'apartment building outdoor',
    'apse indoor',
    'aquarium',
    'aqueduct',
    'arch',
    'archive',
    'arrival gate outdoor',
    'art gallery',
    'art school',
    'art studio',
    'assembly line',
    'athletic field outdoor',
    'atrium public',
    'attic',
    'auditorium',
    'auto factory',
    'badlands',
    'badminton court indoor',
    'baggage claim',
    'bakery shop',
    'balcony exterior',
    'balcony interior',
    'ball pit',
    'ballroom',
    'bamboo forest',
    'banquet hall',
    'bar',
    'barn',
    'barndoor',
    'baseball field',
    'basement',
    'basilica',
    'basketball court outdoor',
    'bathroom',
    'batters box',
    'bayou',
    'bazaar indoor',
    'bazaar outdoor',
    'beach',
    'beauty salon',
    'bedroom',
    'berth',
    'biology laboratory',
    'bistro indoor',
    'boardwalk',
    'boat deck',
    'boathouse',
    'bookstore',
    'booth indoor',
    'botanical garden',
    'bow window indoor',
    'bow window outdoor',
    'bowling alley',
    'boxing ring',
    'brewery indoor',
    'bridge',
    'building facade',
    'bullring',
    'burial chamber',
    'bus interior',
    'butchers shop',
    'butte',
    'cabin outdoor',
    'cafeteria',
    'campsite',
    'campus',
    'canal natural',
    'canal urban',
    'candy store',
    'canyon',
    'car interior backseat',
    'car interior frontseat',
    'carrousel',
    'casino indoor',
    'castle',
    'catacomb',
    'cathedral indoor',
    'cathedral outdoor',
    'cavern indoor',
    'cemetery',
    'chalet',
    'cheese factory',
    'chemistry lab',
    'chicken coop indoor',
    'chicken coop outdoor',
    'childs room',
    'church indoor',
    'church outdoor',
    'classroom',
    'clean room',
    'cliff',
    'cloister indoor',
    'closet',
    'clothing store',
    'coast',
    'cockpit',
    'coffee shop',
    'computer room',
    'conference center',
    'conference room',
    'construction site',
    'control room',
    'control tower outdoor',
    'corn field',
    'corral',
    'corridor',
    'cottage garden',
    'courthouse',
    'courtroom',
    'courtyard',
    'covered bridge exterior',
    'creek',
    'crevasse',
    'crosswalk',
    'cubicle office',
    'dam',
    'delicatessen',
    'dentists office',
    'desert sand',
    'desert vegetation',
    'diner indoor',
    'diner outdoor',
    'dinette home',
    'dinette vehicle',
    'dining car',
    'dining room',
    'discotheque',
    'dock',
    'doorway outdoor',
    'dorm room',
    'driveway',
    'driving range outdoor',
    'drugstore',
    'electrical substation',
    'elevator door',
    'elevator interior',
    'elevator shaft',
    'engine room',
    'escalator indoor',
    'excavation',
    'factory indoor',
    'fairway',
    'fastfood restaurant',
    'field cultivated',
    'field wild',
    'fire escape',
    'fire station',
    'firing range indoor',
    'fishpond',
    'florist shop indoor',
    'food court',
    'forest broadleaf',
    'forest needleleaf',
    'forest path',
    'forest road',
    'formal garden',
    'fountain',
    'galley',
    'game room',
    'garage indoor',
    'garbage dump',
    'gas station',
    'gazebo exterior',
    'general store indoor',
    'general store outdoor',
    'gift shop',
    'golf course',
    'greenhouse indoor',
    'greenhouse outdoor',
    'gymnasium indoor',
    'hangar indoor',
    'hangar outdoor',
    'harbor',
    'hayfield',
    'heliport',
    'herb garden',
    'highway',
    'hill',
    'home office',
    'hospital',
    'hospital room',
    'hot spring',
    'hot tub outdoor',
    'hotel outdoor',
    'hotel room',
    'house',
    'hunting lodge outdoor',
    'ice cream parlor',
    'ice floe',
    'ice shelf',
    'ice skating rink indoor',
    'ice skating rink outdoor',
    'iceberg',
    'igloo',
    'industrial area',
    'inn outdoor',
    'islet',
    'jacuzzi indoor',
    'jail cell',
    'jail indoor',
    'jewelry shop',
    'kasbah',
    'kennel indoor',
    'kennel outdoor',
    'kindergarden classroom',
    'kitchen',
    'kitchenette',
    'labyrinth outdoor',
    'lake natural',
    'landfill',
    'landing deck',
    'laundromat',
    'lecture room',
    'library indoor',
    'library outdoor',
    'lido deck outdoor',
    'lift bridge',
    'lighthouse',
    'limousine interior',
    'living room',
    'lobby',
    'lock chamber',
    'locker room',
    'mansion',
    'manufactured home',
    'market indoor',
    'market outdoor',
    'marsh',
    'martial arts gym',
    'mausoleum',
    'medina',
    'moat water',
    'monastery outdoor',
    'mosque indoor',
    'mosque outdoor',
    'motel',
    'mountain',
    'mountain snowy',
    'movie theater indoor',
    'museum indoor',
    'music store',
    'music studio',
    'nuclear power plant outdoor',
    'nursery',
    'oast house',
    'observatory outdoor',
    'ocean',
    'office',
    'office building',
    'oil refinery outdoor',
    'oilrig',
    'operating room',
    'orchard',
    'outhouse outdoor',
    'pagoda',
    'palace',
    'pantry',
    'park',
    'parking garage indoor',
    'parking garage outdoor',
    'parking lot',
    'parlor',
    'pasture',
    'patio',
    'pavilion',
    'pharmacy',
    'phone booth',
    'physics laboratory',
    'picnic area',
    'pilothouse indoor',
    'planetarium outdoor',
    'playground',
    'playroom',
    'plaza',
    'podium indoor',
    'podium outdoor',
    'pond',
    'poolroom establishment',
    'poolroom home',
    'power plant outdoor',
    'promenade deck',
    'pub indoor',
    'pulpit',
    'putting green',
    'racecourse',
    'raceway',
    'raft',
    'railroad track',
    'rainforest',
    'reception',
    'recreation room',
    'residential neighborhood',
    'restaurant',
    'restaurant kitchen',
    'restaurant patio',
    'rice paddy',
    'riding arena',
    'river',
    'rock arch',
    'rope bridge',
    'ruin',
    'runway',
    'sandbar',
    'sandbox',
    'sauna',
    'schoolhouse',
    'sea cliff',
    'server room',
    'shed',
    'shoe shop',
    'shopfront',
    'shopping mall indoor',
    'shower',
    'skatepark',
    'ski lodge',
    'ski resort',
    'ski slope',
    'sky',
    'skyscraper',
    'slum',
    'snowfield',
    'squash court',
    'stable',
    'stadium baseball',
    'stadium football',
    'stage indoor',
    'staircase',
    'street',
    'subway interior',
    'subway station platform',
    'supermarket',
    'sushi bar',
    'swamp',
    'swimming pool indoor',
    'swimming pool outdoor',
    'synagogue indoor',
    'synagogue outdoor',
    'television studio',
    'temple east asia',
    'temple south asia',
    'tennis court indoor',
    'tennis court outdoor',
    'tent outdoor',
    'theater indoor procenium',
    'theater indoor seats',
    'thriftshop',
    'throne room',
    'ticket booth',
    'toll plaza',
    'topiary garden',
    'tower',
    'toyshop',
    'track outdoor',
    'train railway',
    'train station platform',
    'tree farm',
    'tree house',
    'trench',
    'underwater coral reef',
    'utility room',
    'valley',
    'van interior',
    'vegetable garden',
    'veranda',
    'veterinarians office',
    'viaduct',
    'videostore',
    'village',
    'vineyard',
    'volcano',
    'volleyball court indoor',
    'volleyball court outdoor',
    'waiting room',
    'warehouse indoor',
    'water tower',
    'waterfall block',
    'waterfall fan',
    'waterfall plunge',
    'watering hole',
    'wave',
    'wet bar',
    'wheat field',
    'wind farm',
    'windmill',
    'wine cellar barrel storage',
    'wine cellar bottle storage',
    'wrestling ring indoor',
    'yard',
    'youth hostel',
]



class SUN397(TorchDataset):

    def __init__(self, data_root, tfm,task_id,mode,class_per_task=10, shot=5, B=7, GD_path = '') -> None:

        self.tfm = tfm

        root = os.path.join(data_root,'SUN397')

        image_root = os.path.join(root,'images')

        class_name_file = os.path.join(root,'split/ClassName.txt')
        train_list_file = os.path.join(root,'split/Training_01.txt')
        test_list_file = os.path.join(root,'split/Testing_01.txt')
        with open(class_name_file,'r') as f:
            class_name_list = f.readlines()
            for i in range(len(class_name_list)):
                class_name_list[i] = class_name_list[i].replace('\n','')

        self.class_name_to_id = {}
        for i in range(len(class_name_list)):
            self.class_name_to_id[class_name_list[i]] = i
       
        self.mode = mode

        if self.mode=='train':
            with open(GD_path,'rb') as f:
                self.GD = pickle.load(f)
        self.all_class_num = 397
        self.base_class_num=197

        self.class_per_task = class_per_task
        self.novel_task_len = int((self.all_class_num-self.base_class_num)/class_per_task)
        self.task_len = self.novel_task_len+1
        self.B = B
        self.task_id = task_id
        

        self.images_list = []
        self.labeled_list = []

        class_to_image_dict = {x:[] for x in range(397)}

        if mode=='train':

            task_split = [[] for x in range(self.task_len)]

            for i in range(self.base_class_num):
                task_split[0].append(i)

            for i in range(1,self.task_len):
                for j in range(self.class_per_task):
                    task_split[i].append(self.base_class_num+(i-1)*self.class_per_task+j)

            select_class_id = task_split[task_id]

            self.end_class_id = select_class_id[-1]

        
            with open(train_list_file,'r') as f:
                train_list_all = f.readlines()
                for i in range(len(train_list_all)):
                    train_list_all[i] = train_list_all[i].replace('\n','')
                    class_list_split = train_list_all[i].split("/")[:-1]
                    class_name = "/".join(class_list_split)
                    class_id = self.class_name_to_id[class_name]
                    class_to_image_dict[class_id].append(train_list_all[i])
        


            for c in select_class_id:
                image_path_list = class_to_image_dict[c]

                if c>=self.base_class_num:
                    
                    image_path_list = random.sample(image_path_list,shot)
                for image_path in image_path_list:
                    image_path_full = os.path.join(image_root,image_path[1:])
                  
                    self.images_list.append(image_path_full)
                    self.labeled_list.append(c)
      
  

            self.shot = shot

            self.len = len(self.images_list)


        else:

            self.data = []
            task_to_id_end = {0:self.base_class_num}
            start = self.base_class_num+self.class_per_task
            for i in range(1,self.task_len):
                task_to_id_end[i]=start
                start+=self.class_per_task

            select_class_id=[x for x in range(task_to_id_end[task_id])]

            with open(test_list_file,'r') as f:
                test_list_all = f.readlines()
                for i in range(len(test_list_all)):
                    test_list_all[i] = test_list_all[i].replace('\n','')
                    class_list_split = test_list_all[i].split("/")[:-1]
                    class_name = "/".join(class_list_split)
                    class_id = self.class_name_to_id[class_name]
                    class_to_image_dict[class_id].append(test_list_all[i])


            for c in select_class_id:
                image_path_list = class_to_image_dict[c]

                for image_path in image_path_list:
                    image_path_full = os.path.join(image_root,image_path[1:])
                    self.images_list.append(image_path_full)
                    self.labeled_list.append(c)

            self.shot = shot

            self.len = len(self.images_list)



    

    def __getitem__(self,idx):

        img_name = self.images_list[idx]
        label = self.labeled_list[idx]

        image = Image.open(img_name).convert('RGB')

        if self.tfm:
            image = self.tfm(image)

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
            
            return image,label,pseudo_feat_list,pseudo_label
        return image,label



    def __len__(self):
        return self.len


