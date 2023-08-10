
import os
import torch
import torch.nn as nn
import cv2
from matplotlib import pyplot as plt
import numpy as np

outdir = "dir"
## to check channel feature to compare similarity visually
def save_visualize(each_class, DAP):
    #channel-wise image save
    outdir = "dir"
    bh, k_shot, channel, w, h = each_class.shape
    for b in range(bh):
        for c in range(channel):
            for shot in range(k_shot):
                img = each_class[b][shot][c].cpu().detach().numpy()
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.savefig(os.path.join(outdir+"sample_channel_"+str(shot+1)+"/b_"+str(b)+"_c_"+ str(c) +"_.png"))
                #plt.show()
                #cv2.imwrite(os.path.join(outdir+"sample_channel_"+str(shot+1)+"/"+ str(c) +"_.png"), each_class[b][shot][c])
            dap = DAP[b][0][c].cpu().detach().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(dap)
            plt.savefig(os.path.join(outdir+"dap_channel/b_"+str(b)+"_c_" + str(c) +"_.png")) #, DAP[b][0][c])
        print("one iteration!")



def select_channel(each_class, DAP):
    # each_class => bh, k_shot, 64, 14, 14
    bh, k_shot, channel, w, h = each_class.shape

    m = nn.AdaptiveAvgPool3d((None, 1, 1)).cuda()

    pool_each = m(each_class.view(bh*k_shot, 64, 14, 14)).view(bh, k_shot, 64, 1, 1) # bh, k_shot 64, 1, 1
    pool_DAP = m(DAP.view(bh, 64, 14, 14)) # bh, 64, 1, 1


    result = torch.FloatTensor(bh, 1, channel, w, h).cuda()
    for b in range(bh):
        for c in range(channel):
            sim = []
            for shot in range(k_shot):
                score = abs(pool_DAP[b][c][0] - pool_each[b][shot][c][0]) # attn_each[b][shot][c][0]) #
                sim.append(score)
                # if more similar.. result[bh][shot][i] = attn_each[b][shot][c]
                #cosine 유사도, 유사할 수록 1에 가까워짐
            idx = sim.index(min(sim))
            result[b][0][c] = each_class[b][idx][c]

    return result

def save_visualize_2(each_class, recon, c, i):
    outdir = "dir"
    channel, w, h = each_class.shape

    for cha in range(channel):

        img = each_class[cha].cpu().detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.savefig(os.path.join(outdir+str(cha)+"_.png"))

        r_img = recon[cha].cpu().detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(r_img)
        plt.savefig(os.path.join(outdir+str(cha)+"_recon.png"))



