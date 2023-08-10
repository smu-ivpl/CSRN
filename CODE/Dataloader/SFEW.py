# %%
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from Preprocessing.normalize import norm
from PIL import Image
import csv
import random


class SFEW(Dataset):

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
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 #transforms.Normalize(
                                                 #    mean=[0.5,0.5,0.5],
                                                 #    std=[0.5,0.5,0.5],
                                                 #   ),
                                                 ])

        self.path = os.path.join(root, mode)  # image path
        filedata, labels = self.load_sfew(root, mode)  # data image path
        self.labels = labels.sort()
        self.data = []
        self.img2label = {}
        filedata = sorted(filedata.items(), reverse=False)  # happy : 0, sad : 1, surprise : 2
        self.list_filedata = filedata
        # self.reverse_filedata = {v:k for k,v in dict_filedata.items()}
        for i, (k, v) in enumerate(dict(filedata).items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]             name(一个 类 是一个【】)
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}        将 label id  进行转换  0，1，2，3……
        self.cls_num = len(self.data)
        self.create_batch(self.batchsz, self.labels)

    def load_sfew(self, path, mode):
        """
        #TRAIN PATH = TRAINDIR
        #TEST PATH = TESTDIR
        :return: {label:[file1, file2 ...]}
        """
        dir = os.path.join(path, mode)
        emotionlist = os.listdir(dir)

        dictLabels = {}

        for label in emotionlist:
            filenames = os.listdir(os.path.join(dir + "/" + label))
            for filename in filenames:  # filename: test_xxx.jpg
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, emotionlist

    def create_batch(self, batchsz, labels):  # idx  from  data[[name1,name2],[]  ]
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
            # selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            # np.random.shuffle(selected_cls)
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
            # random.shuffle(support_x)
            # random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def create_test_batch(self, batchsz, labels):  # idx  from  data[[name1,name2],[]  ]
        """
        create batch for meta-learning TEST.
        LOAD ALL TEST DATASET FOR INFERENCE
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch    batchsz * cls * k_shot   [ [cls * k_shot],[],……]
        self.query_x_batch = []  # query set batch        batchsz * cls * k_query
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            # np.random.shuffle(selected_cls)
            support_x = []
            query_x = []

            for cls in range(self.n_way):
                support_x.append([self.data[cls][0]])
                query_x.append(self.data[cls])
                # select_query = np.random.choice(self.data[cls],100, False).tolist()
                # query_x.append(select_query)

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
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)


        flatten_support_x = [os.path.join(self.root + "images/" + item)
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

        # happy = self.list_filedata[0][1]
        # sad = self.list_filedata[1][1]
        # surprise = self.list_filedata[2][1]

        #     [k for k, v in self.dict_filedata.items() if v == item
        support_y = np.array(list(label_dic.values())).astype(np.int32)

        flatten_query_x = [os.path.join(self.root + "images/" + item)
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


        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)


        for l in range(self.n_way):
            support_y_relative[support_y == l] = l
            query_y_relative[query_y == l] = l
        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = norm(path, self.resize)

            # plt.imshow(support_x[i].permute(1,2,0))
            # plt.show()
        for i, path in enumerate(flatten_query_x):
            query_x[i] = norm(path, self.resize)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

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

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
