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

from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, get_downsampled_images,get_downsampled_images_ori, dice_eval
from utils.loss import Grad3d, BinaryDiceLoss, NccLoss
from utils import getters, setters
import os

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt) #,model_type='best'
    model_saver = getters.getModelSaver(opt)

    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)

    if opt['sim_type'] == 'NCC953':
        criterion_sim_0 = NccLoss(win=[9,9,9])
        criterion_sim_1 = NccLoss(win=[5,5,5])
        criterion_sim_2 = NccLoss(win=[3,3,3])
    elif opt['sim_type'] == 'mse':
        criterion_sim_0 = nn.MSELoss()
        criterion_sim_1 = nn.MSELoss()
        criterion_sim_2 = nn.MSELoss()
    criterion_fea = nn.MSELoss()
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
        loss_fea_all = AverageMeter()
        for idx, data in enumerate(train_loader):
            model.train()
           # adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])
            data = [Variable(t.cuda()) for t in data[:10]]
            # x, x_seg = data[0].float(), data[1].long()
            # y, y_seg = data[2].float(), data[3].long()
            x, x_seg, x_sam = data[0].float().cuda(), data[1].long().cuda(), data[-2].float().cuda()
            y, y_seg, y_sam = data[2].float().cuda(), data[3].long().cuda(), data[-1].float().cuda()
            x_sam = F.interpolate(
                x_sam,
                size=(96, 80, 128),
                mode='trilinear',
                align_corners=False
            )
            y_sam = F.interpolate(
                y_sam,
                size=(96, 80, 128),
                mode='trilinear',
                align_corners=False
            )
            #print(x_sam.shape)
            x_seg_oh = F.one_hot(x_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float().cuda()
            y_seg_oh = F.one_hot(y_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float().cuda()
            #print(x_sam.shape, y_sam.shape)

            if 'dinov3IR_abdomen' in opt['model']:
                xs = get_downsampled_images(x, opt['layer_num'], scale=0.5, mode='trilinear')
                ys = get_downsampled_images(y, opt['layer_num'], scale=0.5, mode='trilinear')
                
                fea_xs = get_downsampled_images_ori(x_sam, opt['layer_num'], scale=0.5, mode='trilinear',n_cs=768)
                fea_ys = get_downsampled_images_ori(y_sam, opt['layer_num'], scale=0.5, mode='trilinear',n_cs=768)


                x_seg_ohs = get_downsampled_images(x_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=opt['num_classes'])
                y_seg_ohs = get_downsampled_images(y_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=opt['num_classes'])

                # int_flows, pos_flows = model(x, y)
                model_output = model(x, y, x_sam, y_sam)
                int_flows, pos_flows = model_output[:2]

                reg_loss = 0
                for i in range(opt['layer_num']+1):
                    reg_loss += criterion_reg(int_flows[i]) / 2**i
                reg_loss *= opt['reg_w']
                loss_reg_all.update(reg_loss.detach().cpu().item(), y.numel())

                sim_loss = 0
                for i in range(opt['layer_num']+1):
                    if i==0:
                        sim_loss += criterion_sim_0(model.transformers[i](xs[i], pos_flows[i]), ys[i]) / 2**i
                    elif i==1:
                        sim_loss += criterion_sim_1(model.transformers[i](xs[i], pos_flows[i]), ys[i]) / 2**i
                    else:
                        sim_loss += criterion_sim_2(model.transformers[i](xs[i], pos_flows[i]), ys[i]) / 2**i
                sim_loss *= opt['sim_w']
                loss_sim_all.update(sim_loss.detach().cpu().item(), y.numel())

                dsc_loss = 0
                if opt['dsc_w'] == 0:
                    dsc_loss = reg_loss * 0
                else:
                    for i in range(opt['layer_num']+1):
                        dsc_loss += criterion_dsc(model.transformers[i](x_seg_ohs[i], pos_flows[i]), y_seg_ohs[i]) / 2**i
                    dsc_loss *= opt['dsc_w']
                loss_dsc_all.update(dsc_loss.detach().cpu().item(), y.numel())
                
                ###########
                fea_loss = 0
                if opt['fea_w'] == 0:
                    fea_loss = reg_loss * 0
                else:
                    for i in range(opt['layer_num']+1):
                        fea_loss += criterion_fea(model.transformers[i](fea_xs[i], pos_flows[i]), fea_ys[i]) / 2**i
                    fea_loss *= opt['fea_w']
                loss_fea_all.update(fea_loss.detach().cpu().item(), y.numel())
            else:
                int_flows, pos_flows = model(x, y)
                reg_loss = criterion_reg(int_flows)

                X_Y = model.transformer(x, pos_flows)
                warped_x_seg= model.transformer(x_seg_oh.cuda(), pos_flows)
                
                sim_loss = criterion_sim_0(y, X_Y)
                dsc_loss = criterion_dsc(warped_x_seg, y_seg_oh.cuda())

                loss_reg_all.update(reg_loss.detach().cpu().item(), y.numel())
                loss_sim_all.update(sim_loss.detach().cpu().item(), y.numel())
                loss_dsc_all.update(dsc_loss.detach().cpu().item(), y.numel())

            loss = sim_loss + reg_loss + dsc_loss + fea_loss

            loss_all.update(loss.detach().cpu().item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, FEA {:.4f}'.format(idx, len(train_loader), loss.item(), sim_loss.item(), reg_loss.item(), dsc_loss.item(), fea_loss.item()), end='\r', flush=True)

        print('Epoch [{}/{}], Time {:.2f}, Loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, FEA {:.4f}'.format(epoch, opt['epochs'], time.time()-time_epoch, loss_all.avg, loss_sim_all.avg, loss_reg_all.avg, loss_dsc_all.avg, loss_fea_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        with torch.no_grad():
            for val_idx, data in enumerate(val_loader):
                model.eval()
                data = [Variable(t.cuda()) for t in data[:10]]
                # x, x_seg = data[0].float(), data[1].long()
                # y, y_seg = data[2].float(), data[3].long()
                x, x_seg, x_sam = data[0].float(), data[1].long(), data[-2].float()
                y, y_seg, y_sam = data[2].float(), data[3].long(), data[-1].float()
                x_sam = F.interpolate(
                    x_sam,
                    size=(96, 80, 128),
                    mode='trilinear',
                    align_corners=False
                )
                y_sam = F.interpolate(
                    y_sam,
                    size=(96, 80, 128),
                    mode='trilinear',
                    align_corners=False
                )
                dsc = dice_eval(x_seg.long(), y_seg.long(), opt['num_classes'])
                init_dsc.update(dsc.detach().cpu().item(), x.size(0))

                # x -> y
                # pos_flow = model(x,y,registration=True)
                pos_flow = model(x, y, x_sam, y_sam, registration=True)
                def_out = reg_model(x_seg.float(), pos_flow)
                
                dsc = dice_eval(def_out.long(), y_seg.long(), opt['num_classes'])
                eval_dsc.update(dsc.detach().cpu().item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch

        print('Epoch [{}/{}], Time {:.4f}, init DSC {:.4f}, eval DSC {:.4f}, best DSC {:.4f} at epoch {}'.format(epoch, opt['epochs'], time.time()-time_epoch, init_dsc.avg, eval_dsc.avg, best_dsc, best_epoch))

        model_saver.saveModel(model, epoch, eval_dsc.avg)

if __name__ == '__main__':

    opt = {
        'img_size': (128, 128, 16),  # input image size
        'logs_path': './logs',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 10,          # number of checkpoints to keep
        'power': 0.9,                # decay power
    }

    parser = argparse.ArgumentParser(description="Train FMFReg on AbdomenReg")
    parser.add_argument("-m", "--model", type=str, default='dinov3IR_abdomen')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'abdomenreg')
    parser.add_argument("-dp", "--datasets_path", type=str, default='./data')
    parser.add_argument("--features_path", type=str, default=None,
                        help="Precomputed .npy feature directory; default: <dataset>/features")
    parser.add_argument("--epochs", type = int, default = 301)
    parser.add_argument("--sim_w", type = float, default = 1.)
    parser.add_argument("--reg_w", type = float, default = 1.)
    parser.add_argument("--dsc_w", type = float, default = 1.)
    parser.add_argument("--fea_w", type = float, default = 1.) #1
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--num_workers", type = int, default = 0) # best, last or epoch
    parser.add_argument("--img_size", type=str, default='(96, 80, 128)')
    parser.add_argument("--layer_num", type = int, default = 4)
    parser.add_argument("--sim_type", type = str, default = 'NCC953') # mse or ncc999, ncc995, ncc993 or others
    parser.add_argument("--num_classes", type = int, default = 14) #  number of anatomical classes, 4 for cardiac, 14 for abdominal
    parser.add_argument("--is_bidir", type = int, default = 0) #  only intra-subject cardiac needs bidirectional
    parser.add_argument("--is_hadamard", type = int, default = 1) #  only intra-subject cardiac needs bidirectional

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    from utils.functions import parse_model_kwargs
    opt['nkwargs'] = parse_model_kwargs(unknowns)
    from utils.functions import parse_shape
    opt['img_size'] = parse_shape(opt['img_size'])
    opt['nkwargs']['img_size'] = repr(opt['img_size'])
    opt['nkwargs']['layer_num'] = str(opt['layer_num'])
    opt['nkwargs']['is_hadamard'] = str(opt['is_hadamard'])

    print('sim_w: %.4f, reg_w: %.4f, dsc_w: %.4f, img_size: %s, sim_type: %s, num_classes: %d' % (opt['sim_w'], opt['reg_w'], opt['dsc_w'], opt['img_size'], opt['sim_type'], opt['num_classes']))

    run(opt)
