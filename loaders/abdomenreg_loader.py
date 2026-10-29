import os
import torch
import itertools
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class abdomenreg_loader(Dataset):

    def __init__(self,
            root_dir = './data/abdomenreg',
            split = 'train', # train, val or test
            clips = [-500, 800],
            features_path = None,
            cache_path = None,
        ):

        self.root_dir = root_dir
        self.split = split
        self.clips = clips

        if self.split == 'train':
            idxs = np.arange(1,21)
        elif self.split == 'val':
            idxs = np.arange(21,24)
        elif self.split == 'test':
            idxs = np.arange(24,31)

        img_fps = [os.path.join(root_dir,'img',"img%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]
        lbl_fps = [os.path.join(root_dir,'label',"label%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]
        features_path = features_path or os.path.join(root_dir, 'features')
        emb_fps = [
            os.path.join(features_path, "img%s.npy" % (str(idx).zfill(4))) for
            idx in idxs]

        save_fp = cache_path or os.path.join(root_dir, 'cache')
        os.makedirs(save_fp, exist_ok=True)
        save_fps = [os.path.join(save_fp,"subject%s.npz" % (str(idx).zfill(4))) for idx in idxs]

        self.save_fps = {idx:save_fp for idx, save_fp in zip(idxs, save_fps)}
        self.img_fps = list(itertools.permutations(img_fps, 2))
        self.lbl_fps = list(itertools.permutations(lbl_fps, 2))
        self.emb_fps = list(itertools.permutations(emb_fps, 2))
        self.sub_idx = list(itertools.permutations(idxs, 2))

        print('----->>>> %s set has %d subjects' % (self.split, len(img_fps)))
        print('----->>>> %s set has %d pairs' % (self.split, len(self.sub_idx)))

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):

        sub_idx1, sub_idx2 = self.sub_idx[idx]
        save_fp1, save_fp2 = self.save_fps[sub_idx1], self.save_fps[sub_idx2]

        img_fp1, img_fp2 = self.img_fps[idx]
        lbl_fp1, lbl_fp2 = self.lbl_fps[idx]
        x_emb_fp, y_emb_fp = self.emb_fps[idx]

        if os.path.exists(save_fp1):
            data = np.load(save_fp1)
            x, x_seg, x_emb = data['img'], data['lbl'], data['autolbl']
        else:
            x = np.array(nib.load(img_fp1).get_fdata(), dtype='float32')
            x_seg = np.array(nib.load(lbl_fp1).get_fdata(), dtype='float32')
            x_emb = np.array(np.load(x_emb_fp), dtype='float32')
            x = np.clip(x, self.clips[0], self.clips[1])
            x = (x - self.clips[0]) / (self.clips[1] - self.clips[0])
            x = zoom(x, (0.5, 0.5, 0.5), order=2)
            x_seg = zoom(x_seg, (0.5, 0.5, 0.5), order=0)
            x = x[None, ...]
            x_seg = x_seg[None, ...]
            x_emb = x_emb[0]
            
            np.savez(save_fp1, img=x, lbl=x_seg, autolbl=x_emb)
            print('----->>>> subject # %d saved to %s' % (sub_idx1, save_fp1))

        if os.path.exists(save_fp2):
            data = np.load(save_fp2)
            y, y_seg, y_emb = data['img'], data['lbl'], data['autolbl']
        else:
            y = np.array(nib.load(img_fp2).get_fdata(), dtype='float32')
            y_seg = np.array(nib.load(lbl_fp2).get_fdata(), dtype='float32')
            y_emb = np.array(np.load(y_emb_fp), dtype='float32')
            y = np.clip(y, self.clips[0], self.clips[1])
            y = (y - self.clips[0]) / (self.clips[1] - self.clips[0])
            y = zoom(y, (0.5, 0.5, 0.5), order=2)
            y_seg = zoom(y_seg, (0.5, 0.5, 0.5), order=0)
            y = y[None, ...]
            y_seg = y_seg[None, ...]
            y_emb = y_emb[0]
            np.savez(save_fp2, img=y, lbl=y_seg, autolbl=y_emb)
            print('----->>>> subject # %d saved to %s' % (sub_idx2, save_fp2))

        x, x_seg, x_emb = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(x_emb)
        y, y_seg, y_emb = torch.from_numpy(y), torch.from_numpy(y_seg), torch.from_numpy(y_emb)

        return x, x_seg, y, y_seg, sub_idx1, sub_idx2, x_emb, y_emb
