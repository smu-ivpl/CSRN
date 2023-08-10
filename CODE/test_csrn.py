#CRN_RAFDB07_CHANNEL0519_relation_network_model_best.pth

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RNet_orign import RelationNetWork
from embed import CNNEncoder
from torch.nn import KLDivLoss
from Dataloader.RAFDB import RAFDB
from Select_channel import select_channel as SC
from Dataloader.SFEW import SFEW
from Dataloader.AFEW import AFEW
import numpy as np
import scipy as sp
import scipy.stats
import time

def main():

    n_way = 3
    k_shot = 3
    k_query = 1  # k_shot
    root = "/home/clkim/FER/LearningToCompare_FSL/datas/RAFDB_2/"
    #root = "/home/clkim/FER/LearningToCompare_FSL/datas/AFEW/images_peak_one_ori/"
    #root = "/home/clkim/FER/LearningToCompare_FSL/datas/AFEW/"
    #root = "/home/clkim/FER/LearningToCompare_FSL/datas/SFEW_0518/"
    #mdfile1 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_RAFDB_0522_feature_encoder_model_best.pth"
    #mdfile2 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_RAFDB_0522_relation_network_model_best.pth"
    #mdfile1 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_RAFDB_0531_1414_feature_encoder_model_best.pth"
    #mdfile2 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_RAFDB_0531_1414_relation_network_model_best.pth"
    mdfile1 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_0803_feature_encoder_model_best.pth"
    mdfile2 = "/home/clkim/FER/Pytorch-Implementation-for-RelationNet/MYMODEL/ckpy/SAP_0803_relation_network_model_best.pth"
    feature_embed = CNNEncoder().cpu()  # .to("cuda:0")
    Relation_score = RelationNetWork(64, 8).cpu()
    checkpoint_encoder = torch.load(mdfile1, map_location='cpu')
    checkpoint_rela = torch.load(mdfile2, map_location='cpu')

    feature_embed.load_state_dict(checkpoint_encoder['feature_embed_state_dict'])
    Relation_score.load_state_dict(checkpoint_rela['Relation_score_state_dict'])
    feature_embed.cuda()
    Relation_score.cuda()

    avg_ang = []
    avg_dis = []
    avg_fear = []

    loss_fn = torch.nn.MSELoss().cuda() #to(device)
    print("model name :", mdfile1)
    with torch.no_grad():
        for i in range(1000):
            n_way = 3
            k_shot = 3
            k_query = 1  # k_shot
            batchsize = 25
            tbatch_size = 25
            iteration = int(tbatch_size / batchsize)
            #afew = AFEW(root, mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=tbatch_size,resize=64)  # 1000
            #db = DataLoader(afew, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=False)

            rafdb = RAFDB(root, mode='test',  n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=tbatch_size, resize=64)  # 1000
            db = DataLoader(rafdb,batch_size=batchsize, shuffle=False, num_workers=4,pin_memory=False)
            #sfew = SFEW(root, mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=1, resize=64)  # 1000
            #db = DataLoader(sfew, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=False)

            t_c_ang = 0
            t_c_dis = 0
            t_c_fear = 0
            total_num = 0
            total_num_ang = 0
            total_num_dis = 0
            total_num_fear = 0

            correct = 0
            total = 0
            for i,batch_test in enumerate(db):
                support_x = Variable(batch_test[0])  # .cuda()
                support_y = Variable(batch_test[1]).cuda()  # to(device)
                query_x = Variable(batch_test[2])  # .cuda()#.to("cuda:1")
                query_y = Variable(batch_test[3]).cuda()  # to(device)#.to("cuda:1")
                support_x_eye = Variable(batch_test[4])  # .cuda()#.to("cuda:1")
                support_x_lip = Variable(batch_test[5])  # .cuda()#.to("cuda:1")
                query_x_eye = Variable(batch_test[6])  # .cuda()#.to("cuda:1")
                query_x_lip = Variable(batch_test[7])  # .cuda()#.to("cuda:1")
                bh, set1, c, h, w = support_x.size()
                set2 = query_x.size(1)

                feature_embed.eval()
                Relation_score.eval()

                support_total = torch.cat((support_x.view(bh * set1, c, h, w), support_x_eye.view(bh * set1, c, h, w)),
                                          dim=0)
                support_total = torch.cat((support_total, support_x_lip.view(bh * set1, c, h, w)), dim=0)
                support_total = support_total.cuda()  # to(device)
                # support_totalxf = feature_embed(support_total).view(bh , set1, 64,19,19)
                support_total_xf = feature_embed(support_total)
                support_xf = support_total_xf[:bh * set1].view(bh, set1, 64, 14, 14)
                support_x_eyef = support_total_xf[bh * set1:bh * set1 * 2].view(bh, set1, 64, 14, 14)
                support_x_lipf = support_total_xf[bh * set1 * 2:].view(bh, set1, 64, 14, 14)

                query_total = torch.cat(
                    (query_x.view(bh * set2, c, h, w), query_x_eye.view(bh * set2, c, h, w)), dim=0)
                query_total = torch.cat((query_total, query_x_lip.view(bh * set2, c, h, w)), dim=0)
                query_total = query_total.cuda()  # to(device)
                query_total_xf = feature_embed(query_total)

                query_xf = query_total_xf[:bh * set2].view(bh, set2, 64, 14, 14)
                query_x_eyef = query_total_xf[bh * set2:bh * set2 * 2].view(bh, set2, 64, 14, 14)
                query_x_lipf = query_total_xf[bh * set2 * 2:].view(bh, set2, 64, 14, 14)

                query_xf_jsd = query_xf
                query_x_eyef_jsd = query_x_eyef
                query_x_lipf_jsd = query_x_lipf

                # print("query_f:", query_xf.size())
                batch, clss, _, _, _ = support_xf.shape
                each_class_1f = []
                each_class_2f = []
                each_class_3f = []
                each_class_1e = []
                each_class_2e = []
                each_class_3e = []
                each_class_1l = []
                each_class_2l = []
                each_class_3l = []
                for b in range(batch):
                    for cc in range(clss):
                        if cc == 0 or cc == 1 or cc == 2:
                            each_class_1f.append(support_xf[b][cc])
                            each_class_1e.append(support_x_eyef[b][cc])
                            each_class_1l.append(support_x_lipf[b][cc])
                        if cc == 3 or cc == 4 or cc == 5:
                            each_class_2f.append(support_xf[b][cc])
                            each_class_2e.append(support_x_eyef[b][cc])
                            each_class_2l.append(support_x_lipf[b][cc])
                        if cc == 6 or cc == 7 or cc == 8:
                            each_class_3f.append(support_xf[b][cc])
                            each_class_3e.append(support_x_eyef[b][cc])
                            each_class_3l.append(support_x_lipf[b][cc])

                each_class_1f = torch.stack(each_class_1f, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_2f = torch.stack(each_class_2f, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_3f = torch.stack(each_class_3f, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_1e = torch.stack(each_class_1e, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_2e = torch.stack(each_class_2e, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_3e = torch.stack(each_class_3e, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_1l = torch.stack(each_class_1l, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_2l = torch.stack(each_class_2l, dim=0).view(bh, k_shot, 64, 14, 14)
                each_class_3l = torch.stack(each_class_3l, dim=0).view(bh, k_shot, 64, 14, 14)

                each_class_1fp = each_class_1f.permute(0, 2, 1, 3, 4)
                each_class_2fp = each_class_2f.permute(0, 2, 1, 3, 4)
                each_class_3fp = each_class_3f.permute(0, 2, 1, 3, 4)
                each_class_1ep = each_class_1e.permute(0, 2, 1, 3, 4)
                each_class_2ep = each_class_2e.permute(0, 2, 1, 3, 4)
                each_class_3ep = each_class_3e.permute(0, 2, 1, 3, 4)
                each_class_1lp = each_class_1l.permute(0, 2, 1, 3, 4)
                each_class_2lp = each_class_2l.permute(0, 2, 1, 3, 4)
                each_class_3lp = each_class_3l.permute(0, 2, 1, 3, 4)

                m = nn.AdaptiveAvgPool3d((1, None, None))

                # m = nn.AvgPool3d(((64,14,14)))
                DAP1 = m(each_class_1fp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP2 = m(each_class_2fp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP3 = m(each_class_3fp).permute(0, 2, 1, 3, 4)
                DAP1e = m(each_class_1ep).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP2e = m(each_class_2ep).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP3e = m(each_class_3ep).permute(0, 2, 1, 3, 4)
                DAP1l = m(each_class_1lp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP2l = m(each_class_2lp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
                DAP3l = m(each_class_3lp).permute(0, 2, 1, 3, 4)  #
                # DAP1s = DAP1.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                # DAP2s = DAP2.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                # DAP3s = DAP3.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP1se = DAP1e.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP2se = DAP2e.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP3se = DAP3e.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP1sl = DAP1l.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP2sl = DAP2l.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                DAP3sl = DAP3l.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                # DAP4 = DAP4.unsqueeze(1)

                each_classes = []
                each_classes.append(each_class_1f)
                each_classes.append(each_class_2f)
                each_classes.append(each_class_3f)
                DAPlist = []
                DAPlist.append(DAP1)
                DAPlist.append(DAP2)
                DAPlist.append(DAP3)

                SC_list = []
                for i in range(n_way):
                    SC_list.append(SC(each_classes[i], DAPlist[i]))

                for i in range(n_way):
                    SC_list[i] = SC_list[i].unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)

                DAP = torch.cat((SC_list[0], SC_list[1], SC_list[2]), dim=1)
                # DAP = DAP.squeeze(2)
                # DAP = torch.cat((DAP1s, DAP2s, DAP3s), dim=1)  # 80, 3, 1, 64, 64
                DAPE = torch.cat((DAP1se, DAP2se, DAP3se), dim=1)  # 80, 3, 1, 64, 64
                DAPL = torch.cat((DAP1sl, DAP2sl, DAP3sl), dim=1)  # 80, 3, 1, 64, 64
                bh, set1, _, _, _, _ = DAP.shape

                # 3x3 support 와 3x1 query를 비교해 주기 위해 맞춰주자
                support_xff = support_xf.unsqueeze(1).expand(bh, set2, 9, 64, 14, 14)
                support_x_eyeff = support_x_eyef.unsqueeze(1).expand(bh, set2, 9, 64, 14, 14)
                support_x_lipff = support_x_lipf.unsqueeze(1).expand(bh, set2, 9, 64, 14, 14)
                # 5, 45, 3, 64, 19, 19
                s_query_xff = query_xf.unsqueeze(2).expand(bh, set2, 9, 64, 14, 14)
                s_query_x_eyeff = query_x_eyef.unsqueeze(2).expand(bh, set2, 9, 64, 14, 14)
                s_query_x_lipff = query_x_lipf.unsqueeze(2).expand(bh, set2, 9, 64, 14, 14)
                # 5, 45, 3, 64, 19, 19
                query_xff = query_xf.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)
                query_x_eyeff = query_x_eyef.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)
                query_x_lipff = query_x_lipf.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)

                comb = torch.cat((support_xff, s_query_xff), dim=3)
                comb_eye = torch.cat((support_x_eyeff, s_query_x_eyeff), dim=3)
                comb_lip = torch.cat((support_x_lipff, s_query_x_lipff), dim=3)

                # bh,set2,set1,2c,h,w -> channel 축으로 concat 해줌.. 그럼 한 레이블에 대해
                comb_dap = torch.cat((DAP, s_query_xff), dim=3)
                comb_dape = torch.cat((DAPE, s_query_x_eyeff), dim=3)
                comb_dapl = torch.cat((DAPL, s_query_x_lipff), dim=3)

                batch, cls, cls2, _, _, _ = query_xff.shape

                score = Relation_score(comb.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(3)
                score_e = Relation_score(comb_eye.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(3)
                score_l = Relation_score(comb_lip.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(3)

                score_dap = Relation_score(comb_dap.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(3)
                score_dape = Relation_score(comb_dape.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(
                    3)
                score_dapl = Relation_score(comb_dapl.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9, 1).squeeze(
                    3)

                support_syf = support_y.unsqueeze(1).expand(bh, set2, 9)
                query_syf = query_y.unsqueeze(2).expand(bh, set2, 9)
                slabel = torch.eq(support_syf, query_syf).float()

                # dap label
                y_bat, img = support_y.shape
                support_yy = []
                for i in range(y_bat):
                    support_yy.append([0, 1, 2])
                support_yy = torch.Tensor(support_yy).cuda()

                score = score+ score_l+ score_e +  score_dap * 0.7
                # score = score_dap

                rn_score_np = score.cpu().data.numpy()  # 转numpy cpu
                pred = []
                # support_y_np = support_yy.cpu().data.numpy()
                support_y_np = support_y.cpu().data.numpy()

                for ii, tb in enumerate(rn_score_np):
                    for jj, tset in enumerate(tb):
                        sim = []
                        for way in range(n_way):
                            sim.append(np.sum(tset[way * k_shot:(way + 1) * k_shot]))

                        idx = np.array(sim).argmax()
                        pred.append(support_y_np[ii, idx * k_shot])

                # 此时的pred.size = [b.set2]
                # print("pred.size=", np.array(pred).shape)

                pred = Variable(torch.from_numpy(np.array(pred).reshape(bh, set2))).cuda()  # to(device)

                # 0, 1 , 2 -> angry, disgust, fear
                    # correct_ang = torch.tensor()
                    # correct_dis = torch.tensor()
                    # correct_fear = torch.tensor()

                batch_n, num_each_size = query_y.shape
                correct_ang = []
                correct_dis = []
                correct_fear = []

                for i in range(batch_n):
                    for j in range(num_each_size):
                        if query_y[i][j] == 0:
                            correct_ang.append(torch.eq(pred[i][j], query_y[i][j]))
                        if query_y[i][j] == 1:
                            correct_dis.append(torch.eq(pred[i][j], query_y[i][j]))
                        if query_y[i][j] == 2:
                            correct_fear.append(torch.eq(pred[i][j], query_y[i][j]))

                for a in range(len(correct_ang)):
                    t_c_ang += correct_ang[a].sum()
                    t_c_dis += correct_dis[a].sum()
                    t_c_fear += correct_fear[a].sum()

                # t_c_ang += s_correct_ang.data
                # t_c_dis += s_correct_dis.data
                # t_c_fear += s_correct_fear.data

                correct = torch.eq(pred, query_y).sum()
                total_num += query_y.size(0) * query_y.size(1)
                total_num_ang += len(correct_ang)
                total_num_dis += len(correct_dis)
                total_num_fear += len(correct_fear)

                angry_acc = t_c_ang / total_num_ang
                disgust_acc = t_c_dis / total_num_dis
                fear_acc = t_c_fear / total_num_fear

                accuracy = correct/total
                print("epoch", id, "acc:", accuracy)
                print("correct angry", t_c_ang, "correct disgust", t_c_dis, "correct fear", t_c_fear)
                print("total num angry, disgust, fear : ", total_num_ang, total_num_dis, total_num_fear)
                print("total correct", correct)
                print("total num of 3 emotions: ", total_num)
                print("angry acc:", angry_acc, "disgust acc:", disgust_acc, "fear acc:", fear_acc)

                avg_ang.append(angry_acc)
                avg_dis.append(disgust_acc)
                avg_fear.append(fear_acc)

        print("b a : ",max(avg_ang))
        print("b d : ",max(avg_dis))
        print("b f : ",max(avg_fear))
        #ang_index = avg_ang.index(max(avg_ang))
        #dis_index = avg_dis.index(max(avg_dis))
        #fear_index = avg_fear.index(max(avg_fear))
        #print("ang_index : ", ang_index)
        #print("dis_index : ", dis_index)
        #print("fear_index : ", fear_index)
        print("%%%%%%%%%%%%%%%%%%%%%% AVG TEST ACC %%%%%%%%%%%%%%%%%%%%%%")
        print("angry acc:", sum(avg_ang) / len(avg_ang), "disgust acc:", sum(avg_dis) / len(avg_dis), "fear acc:",
              sum(avg_fear) / len(avg_fear))
        print("done")


if __name__ == '__main__':
    main()





