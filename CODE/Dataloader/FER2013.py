import os
import torch
from torch.utils.data import Dataset
import numpy as np
from Preprocessing.normalize import norm

class FER2013(Dataset):

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.root = root
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        self.path = os.path.join(root, mode)  # image path
        filedata, labels = self.load_fer2013(root, mode) #data image path
        self.labels = labels.sort()
        self.data = []
        self.img2label = {}
        filedata = sorted(filedata.items(), reverse=False)  # happy : 0, sad : 1, surprise : 2
        self.list_filedata = filedata
        #self.reverse_filedata = {v:k for k,v in dict_filedata.items()}
        for i, (k, v) in enumerate(dict(filedata).items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]             name(一个 类 是一个【】)
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}        将 label id  进行转换  0，1，2，3……
        self.cls_num = len(self.data)

        if mode == "train":
            self.create_batch(self.batchsz, self.labels)
        else:
            self.create_val_batch(self.batchsz, self.labels)

    def load_fer2013(self, path, mode):
        """
        #TRAIN PATH = train
        #TEST PATH = test
        :return: {label:[file1, file2 ...]}
        """
        dir = os.path.join(path, mode)
        emotionlist = os.listdir(dir)

        dictLabels = {}

        for label in emotionlist:
            filenames = os.listdir(os.path.join(dir+"/"+label))
            for filename in filenames: #filename: test_xxx.jpg
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, emotionlist

    def create_batch(self, batchsz, labels):                       # idx  from  data[[name1,name2],[]  ]
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """

        self.support_x_batch = []  # support set batch    batchsz * cls * k_shot   [ [cls * k_shot],[],……]
        self.query_x_batch = []  # query set batch        batchsz * cls * k_query
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            #np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in range(4):
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images " filename "  for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            #random.shuffle(support_x)
            #random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets


    def create_val_batch(self, batchsz, labels):                       # idx  from  data[[name1,name2],[]  ]
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch    batchsz * cls * k_shot   [ [cls * k_shot],[],……]
        self.query_x_batch = []  # query set batch        batchsz * cls * k_query
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            #np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in range(self.n_way):
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images " filename "  for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            #random.shuffle(support_x)
            #random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets




    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        #support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        #query_y = np.zeros((self.querysz), dtype=np.int)


        flatten_support_x = [os.path.join(self.root +"images/"+item)
                             for sublist in self.support_x_batch[index] for item in sublist]


        flatten_support_x_eye = [os.path.join(self.root +"images_eye/"+item)
                             for sublist in self.support_x_batch[index] for item in sublist]


        flatten_support_x_lip = [os.path.join(self.root +"images_lip/"+item)
                             for sublist in self.support_x_batch[index] for item in sublist]


        label_dic = {}
        for sublist in self.support_x_batch[index]:
            for item in sublist:
                for i in range(self.n_way):
                    if item in self.list_filedata[i][1]:
                        label_dic[item] = i

        #happy = self.list_filedata[0][1]
        #sad = self.list_filedata[1][1]
        #surprise = self.list_filedata[2][1]

        #     [k for k, v in self.dict_filedata.items() if v == item
        support_y = np.array(list(label_dic.values())).astype(np.int32)

        flatten_query_x = [os.path.join(self.root + "images/" +item)
                           for sublist in self.query_x_batch[index] for item in sublist]

        flatten_query_x_eye = [os.path.join(self.root + "images_eye/" +item)
                           for sublist in self.query_x_batch[index] for item in sublist]

        flatten_query_x_lip = [os.path.join(self.root + "images_lip/" +item)
                           for sublist in self.query_x_batch[index] for item in sublist]

        q_label_dic = {}
        for sublist in self.query_x_batch[index]:
            for item in sublist:
                for i in range(self.n_way):
                    if item in self.list_filedata[i][1]:
                        q_label_dic[item] = i

        query_y = np.array(list(q_label_dic.values())).astype(np.int32)


        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)


        for l in range(self.n_way):
            support_y_relative[support_y==l] = l
            query_y_relative[query_y==l] = l

        # print('relative:', support_y_relative, query_y_relative)


        for i, path in enumerate(flatten_support_x):
            support_x[i] = norm(path, self.resize)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = norm(path, self.resize)
        ########### add cropp support and query #######################

        # if you want to add all eyes, lips and faces together to a set..
        """
        convert_tensor = transforms.ToTensor()
        support_y_relative_total = np.append(support_y_relative, support_y_relative, axis=0)
        support_y_relative_total = np.append(support_y_relative_total, support_y_relative, axis=0)

        query_y_relative_total = np.append(query_y_relative, query_y_relative, axis=0)
        query_y_relative_total = np.append(query_y_relative_total, query_y_relative, axis=0)

        extra_support_x = np.zeros(shape=(8,3,84,84), dtype=np.float32)
        total_support_x = np.append(support_x, extra_support_x, axis=0)
        total_support_x = torch.Tensor(total_support_x)

        extra_query_x = np.zeros(shape=(40, 3, 84, 84), dtype=np.float32)
        total_query_x = np.append(query_x, extra_query_x, axis=0)
        total_query_x = torch.Tensor(total_query_x)
        for i, path in enumerate(flatten_support_x):
            total_support_x[i+s_cnt+1], total_support_x[i+s_cnt+5] = cropping(path, "together")

        for i, path in enumerate(flatten_query_x):
            total_query_x[i+q_cnt+1], total_query_x[i+q_cnt+21] = cropping(path, "together")
            
        return total_support_x, torch.LongTensor(support_y_relative_total), total_query_x, torch.LongTensor(query_y_relative_total)

        """
        support_x_eye = np.zeros(shape=(self.setsz, 3, 64, 64), dtype=np.float32)
        support_x_eye = torch.Tensor(support_x_eye)
        support_x_lip = np.zeros(shape=(self.setsz, 3, 64, 64), dtype=np.float32)
        support_x_lip = torch.Tensor(support_x_lip)

        for i, path in enumerate(flatten_support_x_eye):
            support_x_eye[i] = norm(path, self.resize)

        for i, path in enumerate(flatten_support_x_lip):
            support_x_lip[i] = norm(path, self.resize)


        query_x_eye = np.zeros(shape=(self.querysz, 3, 64, 64), dtype=np.float32)
        query_x_eye = torch.Tensor(query_x_eye)
        query_x_lip = np.zeros(shape=(self.querysz, 3, 64, 64), dtype=np.float32)
        query_x_lip = torch.Tensor(query_x_lip)

        for i, path in enumerate(flatten_query_x_eye):
            query_x_eye[i] = norm(path, self.resize)

        for i, path in enumerate(flatten_query_x_lip):
            query_x_lip[i] = norm(path, self.resize)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative), support_x_eye, support_x_lip, query_x_eye, query_x_lip

        #return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz



















