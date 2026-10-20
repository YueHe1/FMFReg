import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, get_downsampled_images_2D, dice_eval, get_downsampled_images_2D_acdc
from utils.loss import Grad3d, BinaryDiceLoss, NccLoss
from utils import getters, setters

from models.backbones.voxelmorph.torch import layers

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def constancy_rg_loss(Im, If, u):

    Imx, Imy, Imz = torch.gradient(Im, axis=(2,3,4))
    Ifx, Ify, Ifz = torch.gradient(If, axis=(2,3,4))
    Ix = (Imx + Ifx) / 2
    Iy = (Imy + Ify) / 2
    Iz = (Imz + Ifz) / 2
    It = (If - Im) / 2

    Id = Ix*u[:,0:1] + Iy*u[:,1:2] + Iz*u[:,2:3]
    ans = (Id+It)**2

    # grad_mag = torch.sqrt(Ix**2 + Iy**2 + Iz**2 + 1e-5)

    return ans.mean()

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt) # ,model_type='last'
    model_saver = getters.getModelSaver(opt)

    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    transformer_seg = layers.SpatialTransformer(opt['img_size'],'nearest').cuda()
    transformer_img = layers.SpatialTransformer(opt['img_size'],'bilinear').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)

    if opt['sim_type'] == 'NCC953':
        criterion_sim_0 = NccLoss(win=[9,9,9])
        criterion_sim_1 = NccLoss(win=[5,5,5])
        criterion_sim_2 = NccLoss(win=[3,3,3])
    elif opt['sim_type'] == 'mse':
        criterion_sim_0 = nn.MSELoss()
        criterion_sim_1 = nn.MSELoss()
        criterion_sim_2 = nn.MSELoss()
    constancy_rg_0 = nn.MSELoss()
    constancy_rg_1 = nn.MSELoss()
    constancy_rg_2 = nn.MSELoss()
    criterion_reg = Grad3d()
    criterion_dsc = BinaryDiceLoss() # binary dice loss does not require class labels, as all replaced by one-hot encoding already
    best_dsc = 0
    best_epoch = 0

    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        time_epoch = time.time()
        loss_all = AverageMeter()
        loss_sim_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_dsc_all = AverageMeter()
        loss_hsr_all = AverageMeter()
        for idx, data in enumerate(train_loader):
            model.train()
            data = [Variable(t.cuda()) for t in data[:6]]
            x, x_seg, source = data[0].float(), data[1].long(), data[4].float()
            y, y_seg, target = data[2].float(), data[3].long(), data[5].float()
            source = F.interpolate(
                source,
                size=(128, 128, 16),
                mode='trilinear',
                align_corners=False
            )
            target = F.interpolate(
                target,
                size=(128, 128, 16),
                mode='trilinear',
                align_corners=False
            )
            # x, x_seg = data[0].float(), data[1].long()
            # y, y_seg = data[2].float(), data[3].long()
            # subidx = data[4].cpu().numpy().item()
            x_seg_oh = F.one_hot(x_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()
            y_seg_oh = F.one_hot(y_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()

            if ('encoderOnly' in opt['model']) or ('dino' in opt['model']):
                xs = get_downsampled_images_2D_acdc(x, 4, scale=(0.5,0.5,1), mode='trilinear')
                ys = get_downsampled_images_2D_acdc(y, 4, scale=(0.5,0.5,1), mode='trilinear')

                x_seg_ohs = get_downsampled_images_2D_acdc(x_seg_oh, 4, scale=(0.5,0.5,1), mode='trilinear', n_cs=opt['num_classes'])
                y_seg_ohs = get_downsampled_images_2D_acdc(y_seg_oh, 4, scale=(0.5,0.5,1), mode='trilinear', n_cs=opt['num_classes'])

                if opt['hsr_w'] == 0:
                    int_flows, pos_flows = model(x, y, source, target)
                    # int_flows, pos_flows = model(x, y)
                else:
                    # int_flows, pos_flows = model(x, y)
                    int_flows, pos_flows = model(x, y, source, target)
                    sources = get_downsampled_images_2D_acdc(source, 4, scale=(0.5,0.5,1), mode='trilinear', n_cs=768)
                    targets = get_downsampled_images_2D_acdc(target, 4, scale=(0.5,0.5,1), mode='trilinear', n_cs=768)

                reg_loss_0 = criterion_reg(int_flows[0])
                reg_loss_1 = criterion_reg(int_flows[1]) / 2
                reg_loss_2 = criterion_reg(int_flows[2]) / 4
                reg_loss_3 = criterion_reg(int_flows[3]) / 8
                reg_loss_4 = criterion_reg(int_flows[4]) / 16
                reg_loss = (reg_loss_0 + reg_loss_1 + reg_loss_2 + reg_loss_3 + reg_loss_4) * opt['reg_w']
                loss_reg_all.update(reg_loss.item(), y.numel())

                sim_loss_0 = criterion_sim_0(model.transformers[0](xs[0], pos_flows[0]), ys[0])
                sim_loss_1 = criterion_sim_1(model.transformers[1](xs[1], pos_flows[1]), ys[1]) / 2
                sim_loss_2 = criterion_sim_2(model.transformers[2](xs[2], pos_flows[2]), ys[2]) / 4
                sim_loss_3 = criterion_sim_2(model.transformers[3](xs[3], pos_flows[3]), ys[3]) / 8
                sim_loss_4 = criterion_sim_2(model.transformers[4](xs[4], pos_flows[4]), ys[4]) / 16
                sim_loss = (sim_loss_0 + sim_loss_1 + sim_loss_2 + sim_loss_3 + sim_loss_4) * opt['sim_w']
                loss_sim_all.update(sim_loss.item(), y.numel())

                if opt['dsc_w'] == 0:
                    dsc_loss = reg_loss * 0
                else:
                    dsc_loss_0 = criterion_dsc(model.transformers[0](x_seg_ohs[0], pos_flows[0]), y_seg_ohs[0])
                    dsc_loss_1 = criterion_dsc(model.transformers[1](x_seg_ohs[1], pos_flows[1]), y_seg_ohs[1]) / 2
                    dsc_loss_2 = criterion_dsc(model.transformers[2](x_seg_ohs[2], pos_flows[2]), y_seg_ohs[2]) / 4
                    dsc_loss_3 = criterion_dsc(model.transformers[3](x_seg_ohs[3], pos_flows[3]), y_seg_ohs[3]) / 8
                    dsc_loss_4 = criterion_dsc(model.transformers[4](x_seg_ohs[4], pos_flows[4]), y_seg_ohs[4]) / 16
                    dsc_loss = (dsc_loss_0 + dsc_loss_1 + dsc_loss_2 + dsc_loss_3 + dsc_loss_4) * opt['dsc_w']
                loss_dsc_all.update(dsc_loss.item(), y.numel())
            else:
                int_flows, pos_flows = model(x, y)

                reg_loss = criterion_reg(int_flows)

                X_Y = transformer_img(x, pos_flows)
                warped_x_seg= transformer_img(x_seg_oh.cuda(), pos_flows)
                
                sim_loss = criterion_sim_0(y, X_Y)
                dsc_loss = criterion_dsc(warped_x_seg, y_seg_oh.cuda())

                loss_reg_all.update(reg_loss.item(), y.numel())
                loss_sim_all.update(sim_loss.item(), y.numel())
                loss_dsc_all.update(dsc_loss.item(), y.numel())
            
            if opt['hsr_w'] == 0:
                hsr_loss = reg_loss * 0
            else:
                hsr_loss_0 = constancy_rg_0(model.transformers[0](sources[0], pos_flows[0]), targets[0])
                hsr_loss_1 = constancy_rg_1(model.transformers[1](sources[1], pos_flows[1]), targets[1]) / 2
                hsr_loss_2 = constancy_rg_2(model.transformers[2](sources[2], pos_flows[2]), targets[2]) / 4
                hsr_loss_3 = constancy_rg_2(model.transformers[3](sources[3], pos_flows[3]), targets[3]) / 8
                hsr_loss_4 = constancy_rg_2(model.transformers[4](sources[4], pos_flows[4]), targets[4]) / 16
                hsr_loss = (hsr_loss_0 + hsr_loss_1 + hsr_loss_2 + hsr_loss_3 + hsr_loss_4) * opt['hsr_w']
            loss_hsr_all.update(hsr_loss.item(), y.numel())

            loss = sim_loss + reg_loss + dsc_loss+ hsr_loss


            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, HSR {:.4f}'.format(idx, len(train_loader), loss.item(), sim_loss.item(), reg_loss.item(), dsc_loss.item(), hsr_loss.item()), end='\r', flush=True)

        print('Epoch [{}/{}], Time {:.2f}, Loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, HSR {:.4f}'.format(epoch, opt['epochs'], time.time()-time_epoch, loss_all.avg, loss_sim_all.avg, loss_reg_all.avg, loss_dsc_all.avg, loss_hsr_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        with torch.no_grad():
            for val_idx, data in enumerate(val_loader):
                model.eval()
                data = [Variable(t.cuda())  for t in data[:6]]
                x, x_seg, source = data[0].float(), data[1].long(), data[4].float()
                y, y_seg, target = data[2].float(), data[3].long(), data[5].float()
                source = F.interpolate(
                    source,
                    size=(128, 128, 16),
                    mode='trilinear',
                    align_corners=False
                )
                target = F.interpolate(
                    target,
                    size=(128, 128, 16),
                    mode='trilinear',
                    align_corners=False
                )

                dsc = dice_eval(x_seg.long(), y_seg.long(), opt['num_classes'])
                init_dsc.update(dsc.item(), x.size(0))

                # x -> y
                # pos_flow = model(x,y, registration=True)
                pos_flow = model(x, y, source, target, registration=True)
                def_out = reg_model(x_seg.float(), pos_flow)
                
                dsc = dice_eval(def_out.long(), y_seg.long(), opt['num_classes'])
                eval_dsc.update(dsc.item(), x.size(0))

                if opt['is_bidir'] == 1:
                    # y -> x
                    pos_flow = model(y,x,target,source,registration=True)
                    # pos_flow = model(y, x, registration=True)
                    def_out = reg_model(y_seg.cuda().float(), pos_flow)
                    dsc = dice_eval(def_out.long(), x_seg.long(), opt['num_classes'])
                    eval_dsc.update(dsc.item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch

        print('Epoch [{}/{}], Time {:.4f}, init DSC {:.4f}, eval DSC {:.4f}, best DSC {:.4f} at epoch {}'.format(epoch, opt['epochs'], time.time()-time_epoch, init_dsc.avg, eval_dsc.avg, best_dsc, best_epoch))

        model_saver.saveModel(model, epoch, eval_dsc.avg)

if __name__ == '__main__':

    opt = {
        'img_size': (128, 128, 16),  # input image size
        'in_shape': (128, 128, 16),  # input image size
        'logs_path': './logs',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 10,          # number of checkpoints to keep
        'power': 0.9,                # decay power
    }

    parser = argparse.ArgumentParser(description="Train Dinov3IR on ACDC")
    parser.add_argument("-m", "--model", type=str, default='dinov3IR_acdc')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'acdcreg')
    parser.add_argument("-dp", "--datasets_path", type=str, default='./data')
    parser.add_argument("--features_path", type=str, default=None,
                        help="Precomputed .npy feature directory; default: <dataset>/features")
    parser.add_argument("--epochs", type = int, default = 301)
    parser.add_argument("--sim_w", type = float, default = 1.)
    parser.add_argument("--reg_w", type = float, default = 1.)
    parser.add_argument("--dsc_w", type = float, default = 0) #0.1
    parser.add_argument("--hsr_w", type = float, default = 1.)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type = str, default = '(128, 128, 16)')
    parser.add_argument("--sim_type", type = str, default = 'NCC953') # mse or ncc999, ncc995, ncc993 or others
    parser.add_argument("--num_classes", type = int, default = 4) #  number of anatomical classes, 4 for cardiac, 14 for abdominal
    parser.add_argument("--is_bidir", type = int, default = 1) #  only intra-subject cardiac needs bidirectional

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    from utils.functions import parse_model_kwargs
    opt['nkwargs'] = parse_model_kwargs(unknowns)
    from utils.functions import parse_shape
    opt['img_size'] = parse_shape(opt['img_size'])
    opt['nkwargs']['img_size'] = repr(opt['img_size'])

    print('sim_w: %.4f, reg_w: %.4f, dsc_w: %.4f, img_size: %s, sim_type: %s, num_classes: %d' % (opt['sim_w'], opt['reg_w'], opt['dsc_w'], opt['img_size'], opt['sim_type'], opt['num_classes']))

    run(opt)
