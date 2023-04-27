from glob import glob
import numpy as np
import matplotlib.pyplot as plt

spk_list = glob('./conv_feat/*')
for spk in spk_list:
    spk_id = spk.split('/')[-1]
    feats = []
    feat_list = glob(f'{spk}/*.npy')
    
    static_file = open(f'{spk}/{spk_id}_statistic.txt', 'w')
    for feat in feat_list:
        feat = np.load(feat)
        feats.append(feat)

    feats_all = feats[0]
    for feat in feats:
        feats_all = np.concatenate([feats_all, feat])

    feats_all = feats_all.transpose(1,0)

    for i in range(512):
        mean = feats_all[i].mean()
        std = feats_all[i].std()
        print(mean, std)
    '''
    for i in range(512):
        plt.hist(feats_all[i], bins=500)
        plt.savefig(f'./conv_feat/dim_{i}.pdf')
        plt.close()
    '''
