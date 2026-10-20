import os
import torch
import random
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset

class acdcreg_loader(Dataset):

    train_list = [idx for idx in range(1, 17+1)] + [idx for idx in range(21, 37+1)] \
               + [idx for idx in range(41, 57+1)] + [idx for idx in range(61, 77+1)] \
               + [idx for idx in range(81, 97+1)]

    val_list = [idx for idx in range(18, 21)] + [idx for idx in range(38, 41)] \
             + [idx for idx in range(58, 61)] + [idx for idx in range(78, 81)] \
             + [idx for idx in range(98, 100+1)]

    test_list = [idx for idx in range(101, 150+1)]

    def __init__(self,
            root_dir = './data/acdcreg',
            split = 'train', # train, val or test
            enable_random_ed_es_flip = 1,
            features_path = None,
        ):
        self.root_dir = root_dir
        self.split = split
        self.enable_random_ed_es_flip = enable_random_ed_es_flip
        self.emb_path = features_path or os.path.join(root_dir, 'features')

        if self.split == 'train':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.train_list
        elif self.split == 'val':
            self.root_dir = os.path.join(self.root_dir, 'train/')
            self.idxs = self.val_list
        elif self.split == 'test':
            self.root_dir = os.path.join(self.root_dir, 'test/')
            self.idxs = self.test_list
        else:
            raise ValueError('Invalid split name')

        self.init_augmentation()
        self.init_dataset_in_memory()

    def init_augmentation(self):

        self.random_flip = tio.RandomFlip(axes=(0,1))

    def init_dataset_in_memory(self):

        self.data = []
        for sub_idx in self.idxs:
            sub_idx_str = 'patient'+str(sub_idx).zfill(3)
            ed_img_fp = os.path.join(self.root_dir, sub_idx_str+'_ed_img.nii.gz')
            es_img_fp = os.path.join(self.root_dir, sub_idx_str+'_es_img.nii.gz')
            ed_seg_fp = os.path.join(self.root_dir, sub_idx_str+'_ed_seg.nii.gz')
            es_seg_fp = os.path.join(self.root_dir, sub_idx_str+'_es_seg.nii.gz')
            ed_emb_fp = os.path.join(self.emb_path, sub_idx_str + '_ed_img.npy')
            es_emb_fp = os.path.join(self.emb_path, sub_idx_str + '_es_img.npy')

            ed_img = np.asarray(nib.load(ed_img_fp).get_fdata(), dtype=np.float32)
            es_img = np.asarray(nib.load(es_img_fp).get_fdata(), dtype=np.float32)
            ed_seg = np.asarray(nib.load(ed_seg_fp).get_fdata(), dtype=np.float32)
            es_seg = np.asarray(nib.load(es_seg_fp).get_fdata(), dtype=np.float32)
            ed_emb = np.array(np.load(ed_emb_fp), dtype='float32')
            es_emb = np.array(np.load(es_emb_fp), dtype='float32')

            sub = {
                'sub_idx': sub_idx,
                'ed_img': ed_img,
                'es_img': es_img,
                'ed_seg': ed_seg,
                'es_seg': es_seg,
                'ed_emb': ed_emb,
                'es_emb': es_emb,
            }
            self.data.append(sub)
            print(sub_idx)
        print('Number of subjects: ', len(self.data))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        sub = self.data[idx]
        ed_img, es_img, ed_seg, es_seg = sub['ed_img'], sub['es_img'], sub['ed_seg'], sub['es_seg']
        ed_emb = sub['ed_emb']
        es_emb = sub['es_emb']
        x, x_seg, x_emb = ed_img, ed_seg, ed_emb
        y, y_seg, y_emb = es_img, es_seg, es_emb

        if self.enable_random_ed_es_flip and random.random() > 0.5:
            x, y = es_img, ed_img
            x_seg, y_seg = es_seg, ed_seg
            x_emb, y_emb = es_emb, ed_emb

        x, x_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg)
        y, y_seg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg)

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        x, x_seg = x[None,...], x_seg[None,...]
        y, y_seg = y[None,...], y_seg[None,...]
        x_emb, y_emb = x_emb[0], y_emb[0]

        x, x_seg, x_emb = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(x_emb)
        y, y_seg, y_emb = torch.from_numpy(y), torch.from_numpy(y_seg), torch.from_numpy(y_emb)

        if self.split == 'train' and self.enable_random_ed_es_flip:
            subject = tio.Subject(
                img1 = tio.ScalarImage(tensor=x),
                img2 = tio.ScalarImage(tensor=y),
                msk1 = tio.LabelMap(tensor=x_seg),
                msk2 = tio.LabelMap(tensor=y_seg),
                emb1 = tio.LabelMap(tensor=x_emb),
                emb2 = tio.LabelMap(tensor=y_emb),
            )
            subject = self.random_flip(subject)
            x, x_seg, x_emb = subject.img1.data, subject.msk1.data, subject.emb1.data
            y, y_seg, y_emb = subject.img2.data, subject.msk2.data, subject.emb2.data

        return x, x_seg, y, y_seg, x_emb, y_emb, sub['sub_idx']
