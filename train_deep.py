import os
import time
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))
print(working_path)

from utils.loss import weighted_BCE_logits
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from datasets import RS_LEVIR as RS
from models.model_sets.snunet import SNUNet

NET_NAME = ''
DATA_NAME = ''

args = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.02,
    'epochs': 200,
    'gpu': True,
    'lr_decay_power': 1.5,
    'train_crop_size': False,
    'weight_decay': 3e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'predColor_dir': os.path.join(working_path, 'resultColor', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'chkptALL_dir': os.path.join(working_path, 'checkpointsALL', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}

if not os.path.exists(args['log_dir']):
    os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']):
    os.makedirs(args['pred_dir'])
if not os.path.exists(args['predColor_dir']):
    os.makedirs(args['predColor_dir'])
if not os.path.exists(args['chkpt_dir']):
    os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['chkptALL_dir']):
    os.makedirs(args['chkptALL_dir'])

writer = SummaryWriter(args['log_dir'])


def main():
    net = SNUNet().cuda()
    # net1 = LossNet().cuda()
    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=0, shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, optimizer, args, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, optimizer, train_args, val_loader):
    bestaccT = 0
    bestscoreV = 0.0
    bestmIoUV = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])

    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        # net1.eval()
        # for param in net1.parameters():
        #     param.requires_grad = False
        start = time.time()
        acc_meter = AverageMeter()
        f1_meter = AverageMeter()
        train_bn_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)

        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A = data

            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()

            optimizer.zero_grad()

            out_change = net(imgs_A, imgs_B)
            # loss_deep = net1(F.sigmoid(out_change), labels_bn)
            loss_bn = weighted_BCE_logits(out_change, labels_bn)

            loss_bn.backward()
            optimizer.step()

            labels_bn = labels_bn.cpu().detach().numpy()
            preds = F.sigmoid(out_change).cpu().detach() > 0.5

            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels_A):
                acc, valid_sum_A = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_bn_loss.update(loss_bn.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % train_args['print_freq'] == 0:
                print("train")
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [ bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_bn_loss.val, acc_meter.val * 100))  # sc_loss %.4f, train_sc_loss.val,
                writer.add_scalar('train bn_loss', train_bn_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        score_v, mIoU_v, Pre_v, Re_v, acc_v, loss_v = validate(val_loader, net, curr_epoch, train_args)
        if mIoU_v > bestmIoUV:
            bestaccT = acc_meter.avg
            bestmIoUV = mIoU_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_%de_mIoU%.2f_F1 %.2f_OA%.2f.pth' \
                                                  % (curr_epoch, mIoU_v * 100, score_v * 100, acc_v * 100)))
        print("torch.save parameter")
        print('Total time: %.1fs Best rec: Train acc %.2f, Val score %.2f acc %.2f loss %.4f' % (
            time.time() - begin_time, bestaccT * 100, score_v * 100, acc_v * 100, loss_v))
        if score_v > bestscoreV:
            bestscoreV = score_v
            bestaccV = acc_v
            bestloss = loss_v
            print("bestscoreV bestlossV besteraccV")
            print('Total time: %.1fs Best rec: Train acc %.2f, Val score %.2f acc %.2f loss %.4f' % (
                time.time() - begin_time, bestaccT * 100, bestscoreV * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= train_args['epochs']:
            return


def validate(val_loader, net, curr_epoch, train_args):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_bn = labels_A.unsqueeze(1).cuda().float()

        with torch.no_grad():
            out_changes = net(imgs_A, imgs_B)
            loss = weighted_BCE_logits(out_changes, labels_bn)

        val_loss.update(loss.cpu().detach().numpy())
        preds = F.sigmoid(out_changes).cpu().detach() > 0.5
        for (pred, label, label_bn) in zip(preds, labels_A, labels_bn):
            acc, valid_sum = accuracy(pred, label)
            preds_all.append(pred.squeeze().cpu().detach().numpy())
            labels_all.append(label.cpu().detach().numpy())
            acc_meter.update(acc)

    score, IoU_mean, Pre, Re= SCDD_eval_all(preds_all, labels_all, 2)

    curr_time = time.time() - start
    print('val -- %.1fs Val loss: %.2f Score: %.2f mIoU: %.2f Pre: %.2f Re: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), score * 100, IoU_mean * 100, Pre * 100, Re*100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Score', score, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return score, IoU_mean, Pre, Re, acc_meter.avg, val_loss.avg

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def predTrans(files_path=r"",
              outPath=r'/'):

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    files = os.listdir(files_path)
    for file in files:
        file1 = os.path.join(files_path, file)
        img = Image.open(file1)

        img = np.array(img)

        img[img > 0] = 255
        im = Image.fromarray(np.uint8(img))

        im.save(outPath + file.split('\\')[-1])


if __name__ == '__main__':
    main()
