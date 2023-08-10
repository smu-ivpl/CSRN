import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import stepLR
from torch.autograd import Variable
from utils import weight_init
from RNet_orign import RelationNetWork
from embed import CNNEncoder
#from Select_channel import select_channel as SC
from Select_channel import select_channel as SC
from Dataloader.RAFDB import RAFDB
import numpy as np
from logger import Logger
import wandb

LOG_DIR = './log'

root = "" # data dir 
logger = Logger(LOG_DIR)



def main(args):
    wandb.init(project="0803")
    print("0803 abblation study")
    n_way = 4
    k_shot = 4
    k_query = 1 #k_shot
    batchsz = 5
    best_acc = 0

    """
    shuffle DB :train, b:1000, 3-way, 1-shot, 15-query, resize:84
    shuffle DB :test, b:200, 3-way, 1-shot, 15-query, resize:84
    """

    if args.resume:

        feature_embed = CNNEncoder().cpu()  # .to("cuda:0")
        Relation_score = RelationNetWork(64, 8).cpu()
        try:
            checkpoint_encoder = torch.load("modelname.pth", map_location='cpu')
            checkpoint_rela = torch.load("modelname.pth", map_location='cpu')
        except Exception:
            checkpoint_encoder = torch.load("modelname.pth", map_location='cpu')
            checkpoint_rela = torch.load("modelname.pth", map_location='cpu')

        feature_embed.load_state_dict(checkpoint_encoder['feature_embed_state_dict'])
        Relation_score.load_state_dict(checkpoint_rela['Relation_score_state_dict'])
        feature_embed.cuda()
        Relation_score.cuda()

        feature_optim = torch.optim.Adam(feature_embed.parameters(), lr=0.001)
        relation_opim = torch.optim.Adam(Relation_score.parameters(), lr=0.001)
        feature_optim.load_state_dict(checkpoint_encoder['feature_embed_optimizer_state_dict'])
        relation_opim.load_state_dict(checkpoint_rela['Relation_score_optimizer_state_dict'])

        feature_encoder_scheduler = StepLR(feature_optim, step_size=100, gamma=0.5)
        relation_network_scheduler = StepLR(feature_optim, step_size=100, gamma=0.5)

        wandb.watch(feature_embed)
        epochs = 10000 #- checkpoint_encoder['epoch']
        best_acc = checkpoint_encoder['best_acc']
        epoch_cnt = checkpoint_encoder['epoch']

    else:

        feature_embed = CNNEncoder().cuda() # .to("cuda:0")
        Relation_score = RelationNetWork(64, 8).cuda()

        wandb.watch(feature_embed)
        feature_optim = torch.optim.Adam(feature_embed.parameters(), lr=0.001)
        relation_opim = torch.optim.Adam(Relation_score.parameters(), lr=0.001)

        feature_encoder_scheduler = StepLR(feature_optim, step_size=100, gamma=0.5)
        relation_network_scheduler = StepLR(feature_optim, step_size=100, gamma=0.5)
        epochs = 10000
        epoch_cnt = 0



    feature_embed.apply(weight_init)
    Relation_score.apply(weight_init)

    loss_fn = torch.nn.MSELoss().cuda() #to(device)

    anglist=[]
    dislist=[]
    fearlist=[]
    for epoch in range(epoch_cnt+1, epochs):
        feature_encoder_scheduler.step(epoch)
        relation_network_scheduler.step(epoch)
        n_way = 4
        k_shot = 4
        k_query = 1
        #epo = epoch + epoch_cnt
        batchsize = 25
        tbatch_size = 25

        iteration = int(tbatch_size / batchsize)

        rafdb = RAFDB(root, mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=tbatch_size, resize=64) #38400
        db = DataLoader(rafdb,batch_size=batchsize,shuffle=False,num_workers=4,pin_memory=True)  # 64 , 5*(1+15) =  n_way*(k_shot+k_query), , c, h, w

        checkpoint_state = {
            'epoch': epoch,
            'feature_embed_state_dict': feature_embed.state_dict(),
            'Relation_score_state_dict': Relation_score.state_dict(),
            'best_acc': best_acc,
            'feature_embed_optimizer_state_dict': feature_optim.state_dict(),
            'Relation_score_optimizer_state_dict': relation_opim.state_dict(),
        }


        ## accuracy values
        total_correct = 0
        total_num = 0
        accuracy = 0
        t_c_ang = 0
        t_c_dis = 0
        t_c_fear = 0
        total_num_ang = 0
        total_num_dis = 0
        total_num_fear = 0
        angry_acc = 0
        disgust_acc = 0
        fear_acc = 0
        loss_li = []
        for step, batch in enumerate(db):
            support_x = Variable(batch[0]) # [batch_size, n_way*(k_shot+k_query), c , h , w]
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2])
            query_y = Variable(batch[3]).cuda()
            support_x_eye = Variable(batch[4])
            support_x_lip = Variable(batch[5])
            query_x_eye = Variable(batch[6])
            query_x_lip = Variable(batch[7])

            n_way=4
            k_shot=4
            k_query=1

            bh, set1, c, h, w = support_x.size()
            set2 = query_x.size(1)

            feature_embed.train()
            Relation_score.train()
            # wandb.watch(Relation_score)

            support_total = torch.cat((support_x.view(bh * set1, c, h, w), support_x_eye.view(bh * set1, c, h, w)), dim=0)
            support_total = torch.cat((support_total, support_x_lip.view(bh * set1, c, h, w)), dim=0)
            support_total = support_total.cuda()#to(device)
            #support_totalxf = feature_embed(support_total).view(bh , set1, 64,19,19)
            support_total_xf = feature_embed(support_total)
            support_xf = support_total_xf[:bh * set1].view(bh, set1, 64, 14, 14)
            support_x_eyef = support_total_xf[bh * set1:bh * set1*2].view(bh, set1, 64, 14, 14)
            support_x_lipf = support_total_xf[bh * set1*2:].view(bh, set1, 64, 14, 14)

            query_total = torch.cat((query_x.view(bh * set2, c, h, w), query_x_eye.view(bh * set2, c, h, w)), dim=0)
            query_total = torch.cat((query_total, query_x_lip.view(bh * set2, c, h, w)), dim=0)
            query_total = query_total.cuda() #to(device)
            query_total_xf = feature_embed(query_total)

            query_xf = query_total_xf[:bh * set2].view(bh, set2, 64, 14, 14)
            query_x_eyef = query_total_xf[bh * set2:bh * set2*2].view(bh, set2, 64, 14, 14)
            query_x_lipf = query_total_xf[bh * set2*2:].view(bh, set2, 64, 14, 14)

            query_xf_jsd = query_xf
            query_x_eyef_jsd = query_x_eyef
            query_x_lipf_jsd = query_x_lipf

            # print("query_f:", query_xf.size())
            batch, clss, _, _, _ = support_xf.shape
            each_class_1f = []
            each_class_2f = []
            each_class_3f = []
            each_class_4f = []
            each_class_1e = []
            each_class_2e = []
            each_class_3e = []
            each_class_4e = []
            each_class_1l = []
            each_class_2l = []
            each_class_3l = []
            each_class_4l = []
            for b in range(batch):
                for cc in range(clss):
                    if cc == 0 or cc == 1 or cc == 2 or cc == 3:
                        each_class_1f.append(support_xf[b][cc])
                        each_class_1e.append(support_x_eyef[b][cc])
                        each_class_1l.append(support_x_lipf[b][cc])
                    if cc == 4 or cc == 5 or cc == 6 or cc == 7:
                        each_class_2f.append(support_xf[b][cc])
                        each_class_2e.append(support_x_eyef[b][cc])
                        each_class_2l.append(support_x_lipf[b][cc])
                    if cc == 8 or cc == 9 or cc == 10 or cc == 11:
                        each_class_3f.append(support_xf[b][cc])
                        each_class_3e.append(support_x_eyef[b][cc])
                        each_class_3l.append(support_x_lipf[b][cc])
                    if cc == 12 or cc == 13 or cc == 14 or cc == 15:
                        each_class_4f.append(support_xf[b][cc])
                        each_class_4e.append(support_x_eyef[b][cc])
                        each_class_4l.append(support_x_lipf[b][cc])

            each_class_1f = torch.stack(each_class_1f, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_2f = torch.stack(each_class_2f, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_3f = torch.stack(each_class_3f, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_4f = torch.stack(each_class_4f, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_1e = torch.stack(each_class_1e, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_2e = torch.stack(each_class_2e, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_3e = torch.stack(each_class_3e, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_4e = torch.stack(each_class_4e, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_1l = torch.stack(each_class_1l, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_2l = torch.stack(each_class_2l, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_3l = torch.stack(each_class_3l, dim=0).view(bh, k_shot, 64, 14, 14)
            each_class_4l = torch.stack(each_class_4l, dim=0).view(bh, k_shot, 64, 14, 14)

            each_class_1fp = each_class_1f.permute(0, 2, 1, 3, 4)
            each_class_2fp = each_class_2f.permute(0, 2, 1, 3, 4)
            each_class_3fp = each_class_3f.permute(0, 2, 1, 3, 4)
            each_class_4fp = each_class_4f.permute(0, 2, 1, 3, 4)
            each_class_1e = each_class_1e.permute(0, 2, 1, 3, 4)
            each_class_2e = each_class_2e.permute(0, 2, 1, 3, 4)
            each_class_3e = each_class_3e.permute(0, 2, 1, 3, 4)
            each_class_4e = each_class_4e.permute(0, 2, 1, 3, 4)
            each_class_1l = each_class_1l.permute(0, 2, 1, 3, 4)
            each_class_2l = each_class_2l.permute(0, 2, 1, 3, 4)
            each_class_3l = each_class_3l.permute(0, 2, 1, 3, 4)
            each_class_4l = each_class_4l.permute(0, 2, 1, 3, 4)

            m = nn.AdaptiveAvgPool3d((1, None, None))

            # m = nn.AvgPool3d(((64,14,14)))
            DAP1 = m(each_class_1fp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP2 = m(each_class_2fp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP3 = m(each_class_3fp).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP4 = m(each_class_4fp).permute(0, 2, 1, 3, 4)
            DAP1e = m(each_class_1e).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP2e = m(each_class_2e).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP3e = m(each_class_3e).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP4e = m(each_class_4e).permute(0, 2, 1, 3, 4)
            DAP1l = m(each_class_1l).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP2l = m(each_class_2l).permute(0, 2, 1, 3, 4)  # .expand(bh, k_shot, 64, 14, 14)
            DAP3l = m(each_class_3l).permute(0, 2, 1, 3, 4)
            DAP4l = m(each_class_4l).permute(0, 2, 1, 3, 4)
            #DAP1s = DAP1.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            #DAP2s = DAP2.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            #DAP3s = DAP3.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            #DAP4s = DAP4.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP1se = DAP1e.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP2se = DAP2e.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP3se = DAP3e.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP4se = DAP4e.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP1sl = DAP1l.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP2sl = DAP2l.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP3sl = DAP3l.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            DAP4sl = DAP4l.unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)
            # DAP4 = DAP4.unsqueeze(1)


            ### select channel!
            each_classes = []
            each_classes.append(each_class_1f)
            each_classes.append(each_class_2f)
            each_classes.append(each_class_3f)
            each_classes.append(each_class_4f)
            DAPlist = []
            DAPlist.append(DAP1)
            DAPlist.append(DAP2)
            DAPlist.append(DAP3)
            DAPlist.append(DAP4)

            SC_list = []
            for i in range(n_way):
                SC_list.append(SC(each_classes[i], DAPlist[i]))

            for i in range(n_way):
                SC_list[i] = SC_list[i].unsqueeze(1).expand(bh, 1, 4, 64, 14, 14)


            DAP = torch.cat((SC_list[0], SC_list[1], SC_list[2], SC_list[3]),dim=1)


            # original
            #DAP = torch.cat((DAP1s, DAP2s, DAP3s, DAP4s), dim=1)  # 80, 4, 1, 64, 64
            DAPE = torch.cat((DAP1se, DAP2se, DAP3se, DAP4se), dim=1)  # 80, 4, 1, 64, 64
            DAPL = torch.cat((DAP1sl, DAP2sl, DAP3sl, DAP4sl), dim=1)  # 80, 4, 1, 64, 64
            bh, set1, _, _, _, _ = DAP.shape

            # 3x3 support 와 3x1 query를 비교해 주기 위해 맞춰주자
            support_xff = support_xf.unsqueeze(1).expand(bh, set2, 16, 64, 14, 14)
            support_x_eyeff = support_x_eyef.unsqueeze(1).expand(bh, set2, 16, 64, 14, 14)
            support_x_lipff = support_x_lipf.unsqueeze(1).expand(bh, set2, 16, 64, 14, 14)
            # 5, 45, 3, 64, 19, 19
            s_query_xff = query_xf.unsqueeze(2).expand(bh, set2, 16, 64, 14, 14)
            s_query_x_eyeff = query_x_eyef.unsqueeze(2).expand(bh, set2, 16, 64, 14, 14)
            s_query_x_lipff = query_x_lipf.unsqueeze(2).expand(bh, set2, 16, 64, 14, 14)
            # 5, 45, 3, 64, 19, 19
            query_xff = query_xf.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)
            query_x_eyeff = query_x_eyef.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)
            query_x_lipff = query_x_lipf.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)

            batch, cls, cls2, _, _, _ = query_xff.shape
            JSD_class_1 = []
            JSD_class_2 = []
            JSD_class_3 = []
            JSD_class_4 = []
            JSD_class_1e = []
            JSD_class_2e = []
            JSD_class_3e = []
            JSD_class_4e = []
            JSD_class_1l = []
            JSD_class_2l = []
            JSD_class_3l = []
            JSD_class_4l = []
            t_list = []
            t_liste = []
            t_listl = []
            for bb in range(batch):
                for cs in range(cls):
                    if cs == 0:  # or cc == 1 or cc == 2:
                        JSD_class_1 = [abs(JSD(DAP1[bb][0], query_xf_jsd[bb][cs]))]
                        JSD_class_1e = [abs(JSD(DAP1e[bb][0], query_x_eyef_jsd[bb][cs]))]
                        JSD_class_1l = [abs(JSD(DAP1l[bb][0], query_x_lipf_jsd[bb][cs]))]
                    if cs == 1:  # or cc == 4 or cc == 5:
                        JSD_class_2 = [abs(JSD(DAP2[bb][0], query_xf_jsd[bb][cs]))]
                        JSD_class_2e = [abs(JSD(DAP2e[bb][0], query_x_eyef_jsd[bb][cs]))]
                        JSD_class_2l = [abs(JSD(DAP2l[bb][0], query_x_lipf_jsd[bb][cs]))]
                    if cs == 2:  # or cc == 7 or cc == 8:
                        JSD_class_3 = [abs(JSD(DAP3[bb][0], query_xf_jsd[bb][cs]))]
                        JSD_class_3e = [abs(JSD(DAP3e[bb][0], query_x_eyef_jsd[bb][cs]))]
                        JSD_class_3l = [abs(JSD(DAP3l[bb][0], query_x_lipf_jsd[bb][cs]))]
                    if cs == 3:  # or cc == 7 or cc == 8:
                        JSD_class_4 = [abs(JSD(DAP4[bb][0], query_xf_jsd[bb][cs]))]
                        JSD_class_4e = [abs(JSD(DAP4e[bb][0], query_x_eyef_jsd[bb][cs]))]
                        JSD_class_4l = [abs(JSD(DAP4l[bb][0], query_x_lipf_jsd[bb][cs]))]

                JSD_class_1t = torch.stack(JSD_class_1, dim=0)  # batch  사이즈 만큼 생성됨
                JSD_class_2t = torch.stack(JSD_class_2, dim=0)
                JSD_class_3t = torch.stack(JSD_class_3, dim=0)
                JSD_class_4t = torch.stack(JSD_class_4, dim=0)
                JSD_class_1te = torch.stack(JSD_class_1e, dim=0)  # batch  사이즈 만큼 생성됨
                JSD_class_2te = torch.stack(JSD_class_2e, dim=0)
                JSD_class_3te = torch.stack(JSD_class_3e, dim=0)
                JSD_class_4te = torch.stack(JSD_class_4e, dim=0)
                JSD_class_1tl = torch.stack(JSD_class_1l, dim=0)  # batch  사이즈 만큼 생성됨
                JSD_class_2tl = torch.stack(JSD_class_2l, dim=0)
                JSD_class_3tl = torch.stack(JSD_class_3l, dim=0)
                JSD_class_4tl = torch.stack(JSD_class_4l, dim=0)

                # 240, 3   # 80, 3
                jsd_comb = torch.cat((JSD_class_1t, JSD_class_2t, JSD_class_3t, JSD_class_4t), dim=0)
                jsd_comb_e = torch.cat((JSD_class_1te, JSD_class_2te, JSD_class_3te, JSD_class_4te), dim=0)
                jsd_comb_l = torch.cat((JSD_class_1tl, JSD_class_2tl, JSD_class_3tl, JSD_class_4tl), dim=0)
                t_list.append(jsd_comb)
                t_liste.append(jsd_comb_e)
                t_listl.append(jsd_comb_l)

            final_jsd = torch.stack(t_list, dim=0)
            final_jsd_e = torch.stack(t_liste, dim=0)
            final_jsd_l = torch.stack(t_listl, dim=0)

            final_jsd = final_jsd.unsqueeze(1).expand(batchsize, set1, set2)
            final_jsd_e = final_jsd_e.unsqueeze(1).expand(batchsize, set1, set2)
            final_jsd_l = final_jsd_l.unsqueeze(1).expand(batchsize, set1, set2)

            comb = torch.cat((support_xff, s_query_xff), dim=3)
            comb_eye = torch.cat((support_x_eyeff, s_query_x_eyeff), dim=3)
            comb_lip = torch.cat((support_x_lipff, s_query_x_lipff), dim=3)

            # bh,set2,set1,2c,h,w -> channel 축으로 concat 해줌.. 그럼 한 레이블에 대해
            comb_dap = torch.cat((DAP, query_xff), dim=3)
            comb_dape = torch.cat((DAPE, query_x_eyeff), dim=3)
            comb_dapl = torch.cat((DAPL, query_x_lipff), dim=3)

            score = Relation_score(comb.view(bh * set2 * 16, 2 * 64, 14, 14)).view(bh, set2, 16, 1).squeeze(3)
            score_e = Relation_score(comb_eye.view(bh * set2 * 16, 2 * 64, 14, 14)).view(bh, set2, 16, 1).squeeze(3)
            score_l = Relation_score(comb_lip.view(bh * set2 * 16, 2 * 64, 14, 14)).view(bh, set2, 16, 1).squeeze(3)

            score_dap = Relation_score(comb_dap.view(bh * set2 * set1, 2 * 64, 14, 14)).view(bh, set2, set1, 1).squeeze(3)
            score_dape = Relation_score(comb_dape.view(bh * set2 * set1, 2 * 64, 14, 14)).view(bh, set2, set1, 1).squeeze(3)
            score_dapl = Relation_score(comb_dapl.view(bh * set2 * set1, 2 * 64, 14, 14)).view(bh, set2, set1, 1).squeeze(3)

            support_yf = support_y.unsqueeze(1).expand(bh, set2, 16)
            query_yf = query_y.unsqueeze(2).expand(bh, set2, 16)
            slabel = torch.eq(support_yf, query_yf).float()

            y_bat, img = support_y.shape
            support_yy = []
            for i in range(y_bat):
                support_yy.append([0, 1, 2, 4])
            support_yy = torch.Tensor(support_yy).cuda()
            support_yf = support_yy.unsqueeze(1).expand(bh, set2, set1)
            query_yf = query_y.unsqueeze(2).expand(bh, set2, set1)
            label = torch.eq(support_yf, query_yf).float()

            # jsd_label 은 dap 와 동일하게 스면될듯..
            # jsd_label = torch.eq(final_jsd, query_yf).float() # 둘 모두 [000][111]... 순서로 간다.
            feature_optim.zero_grad()
            relation_opim.zero_grad()
            # 0.5 * score  * score_dap + 0.5 * loss_jsd
            lf = loss_fn(score, slabel)
            le = loss_fn(score_e, slabel)
            ll = loss_fn(score_l, slabel)
            loss_dap = loss_fn(score_dap, label)
            loss_dape = loss_fn(score_dape, label)
            loss_dapl = loss_fn(score_dapl, label)
            jsd = loss_fn(final_jsd, label)
            jsde = loss_fn(final_jsd_e, label)
            jsdl = loss_fn(final_jsd_l, label)
            feature_optim.zero_grad()
            relation_opim.zero_grad()

            #  0.3* score + 0.6*score_e + 0.1*score_l +  0.6* score_dape + 0.1*score_dapl + 0.5 * ( 0.6*final_jsd_e +  0.1*final_jsd_l)
            loss = lf  + ll + le + loss_dap * 0.7 # + 0.1*loss_dapl + 0.5 * (0.6*jsde + 0.1*jsdl)
            loss.backward()


            torch.nn.utils.clip_grad_norm(feature_embed.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(Relation_score.parameters(), 0.5)

            feature_optim.step()
            relation_opim.step()

            loss_li.append(loss.data)

            if epoch % 2 == 0:
                print("---------test--------")
                with torch.no_grad():
                    for i in range(15):
                        print("test : ", i)
                        n_way=3
                        k_shot = 3
                        k_query = 1
                        rafdb_val = RAFDB(root, mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=tbatch_size, resize=64)
                        # val_sampler = DistributedSampler(rafdb_val, num_replicas=hvd.size(), rank=hvd.rank())
                        db_val = DataLoader(rafdb_val, batch_size=batchsize, shuffle=False, num_workers=4,pin_memory=True)  # , sampler=val_sampler)

                        for j, batch_test in enumerate(db_val):
                            support_x = Variable(batch_test[0])#.cuda()
                            support_y = Variable(batch_test[1]).cuda() #to(device)
                            query_x = Variable(batch_test[2])#.cuda()#.to("cuda:1")
                            query_y = Variable(batch_test[3]).cuda() #to(device)#.to("cuda:1")
                            support_x_eye = Variable(batch_test[4])#.cuda()#.to("cuda:1")
                            support_x_lip = Variable(batch_test[5])#.cuda()#.to("cuda:1")
                            query_x_eye = Variable(batch_test[6])#.cuda()#.to("cuda:1")
                            query_x_lip = Variable(batch_test[7])#.cuda()#.to("cuda:1")
                            bh, set1, c, h, w = support_x.size()
                            set2 = query_x.size(1)

                            feature_embed.eval()
                            Relation_score.eval()

                            support_total = torch.cat((support_x.view(bh * set1, c, h, w), support_x_eye.view(bh * set1, c, h, w)), dim=0)
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
                            #DAP1s = DAP1.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                            #DAP2s = DAP2.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
                            #DAP3s = DAP3.unsqueeze(1).expand(bh, 1, 9, 64, 14, 14)
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
                            #DAP = torch.cat((DAP1s, DAP2s, DAP3s), dim=1)  # 80, 3, 1, 64, 64
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
                            query_x_eyeff = query_x_eyef.unsqueeze(2).expand(bh, set2,set1, 64, 14, 14)
                            query_x_lipff = query_x_lipf.unsqueeze(2).expand(bh, set2, set1, 64, 14, 14)


                            comb = torch.cat((support_xff, s_query_xff), dim=3)
                            comb_eye = torch.cat((support_x_eyeff, s_query_x_eyeff), dim=3)
                            comb_lip = torch.cat((support_x_lipff, s_query_x_lipff), dim=3)

                            # bh,set2,set1,2c,h,w -> channel 축으로 concat 해줌.. 그럼 한 레이블에 대해
                            comb_dap = torch.cat((DAP, s_query_xff), dim=3)
                            comb_dape = torch.cat((DAPE, s_query_x_eyeff), dim=3)
                            comb_dapl = torch.cat((DAPL, s_query_x_lipff), dim=3)

                            batch, cls, cls2, _, _, _ = query_xff.shape

                            score = Relation_score(comb.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9,1).squeeze(3)
                            score_e = Relation_score(comb_eye.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9,1).squeeze(3)
                            score_l = Relation_score(comb_lip.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9,1).squeeze(3)

                            score_dap = Relation_score(comb_dap.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2,9,1).squeeze(3)
                            score_dape = Relation_score(comb_dape.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9,1).squeeze(3)
                            score_dapl = Relation_score(comb_dapl.view(bh * set2 * 9, 2 * 64, 14, 14)).view(bh, set2, 9,1).squeeze(3)

                            support_syf = support_y.unsqueeze(1).expand(bh, set2, 9)
                            query_syf = query_y.unsqueeze(2).expand(bh, set2, 9)
                            slabel = torch.eq(support_syf, query_syf).float()

                            # dap label
                            y_bat, img = support_y.shape
                            support_yy = []
                            for i in range(y_bat):
                                support_yy.append([0, 1, 2])
                            support_yy = torch.Tensor(support_yy).cuda()

                            #0.5 x l x d + 0.5 j
                            #0.6*lf + 0.2 * ll + 0.2 * le + 0.7 * loss_dap
                            score = score + score_l +  score_e + score_dap * 0.7  #+ 0.1*score_dapl + 0.5 * ( 0.6*final_jsd_e +  0.1*final_jsd_l)
                            # 3x3 support 와 3x1 query label
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
                            total_correct += correct.data  # [0]
                            total_num += query_y.size(0) * query_y.size(1)
                            total_num_ang += len(correct_ang)
                            total_num_dis += len(correct_dis)
                            total_num_fear += len(correct_fear)

                        #del support_x, support_y, query_x, query_y, support_x_eye, support_x_lip, query_x_eye, query_x_lip, support_total_xf, query_total_xf
                        #torch.cuda.empty_cache()

                angry_acc = t_c_ang / total_num_ang
                disgust_acc = t_c_dis / total_num_dis
                fear_acc = t_c_fear / total_num_fear
                accuracy = total_correct / total_num

                logger.log_value('acc : ', accuracy)
                print("epoch:", epoch, "acc:", accuracy)
                print("correct angry", t_c_ang, "correct disgust", t_c_dis, "correct fear", t_c_fear)
                print("total num angry, disgust, fear : ", total_num_ang, total_num_dis, total_num_fear)
                print("total correct", total_correct)
                print("total num of 3 emotions: ", total_num)
                print("angry acc:", angry_acc, "disgust acc:", disgust_acc, "fear acc:", fear_acc)

                wandb.log({
                    "CL - eval accuracy": accuracy,
                    "CL - angry acc": angry_acc,
                    "CL - disgust acc": disgust_acc,
                    "CL - fear acc": fear_acc
                })

            anglist.append(angry_acc)
            dislist.append(disgust_acc)
            fearlist.append(fear_acc)
            if accuracy > best_acc:
                print("-------------------epoch", epoch, "step:", step, "acc:", accuracy,
                      "---------------------------------------")
                best_acc = accuracy
                filename = "feature_model_best.pth"
                torch.save(checkpoint_state, filename, _use_new_zipfile_serialization=False )
                filename2 = "relation_model_best.pth"
                torch.save(checkpoint_state, filename2, _use_new_zipfile_serialization=False )


        logger.log_value('{}-way-{}-shot loss：'.format(n_way, k_shot), loss.data)

        train_loss = sum(loss_li) / iteration

        print("epoch:", epoch, "train_loss: ", train_loss)

        wandb.log({
            "CL - train loss": train_loss,
        })

        last_file = "feature_last_modelname.pth"
        torch.save(checkpoint_state, last_file, _use_new_zipfile_serialization=False)
        last_file2 = "relation_last_modelname.pth"
        torch.save(checkpoint_state, last_file2, _use_new_zipfile_serialization=False)

    angmean = sum(anglist) / len(anglist)
    dismean = sum(dislist) / len(dislist)
    fearmean = sum(fearlist) / len(fearlist)
    print("%%%%%%%%%%%%%%% average eval accuracy %%%%%%%%%%%%%%%")
    print("Average Angry : ", angmean)
    print("Average Disgust : ", dismean)
    print("Average Fear : ", fearmean)
    logger.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', action="store_true", help="resume train", default=False)
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    main(args)
