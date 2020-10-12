from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net as UAN
from data import get_training_set, get_eval_set
import pdb
import socket
import time
import numpy as np
import matplotlib.pyplot as plt
#torch.backends.cudnn.benchmark = True


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--hr_eval_dataset', type=str, default='DIV2K_eval_HR')
parser.add_argument('--model_type', type=str, default='UAN')
parser.add_argument('--patch_size', type=int, default=48, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='OURMODEL_DIV2K_epoch_100.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='_DIV2K', help='Location to save checkpoint models')
parser.add_argument('--psnr_curve', default=True, help='Show PSNR curve and save')
parser.add_argument('--skip_threshold', type=float, default=1e14, help='skipping batch that has large error')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])
        optimizer.zero_grad()
#        for param in model.parameters():
#            param.grad = None
        t0 = time.time()
        prediction = model(input)

        loss = criterionL1(prediction, target)
        
        if loss.item() < opt.skip_threshold:
            loss.backward()
            optimizer.step()
        else:
            print('Skip this batch {}! (Loss: {})'.format(
                iteration + 1, loss.item()
            ))
                
        t1 = time.time()
        epoch_loss += loss.data

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    

def evaluate(epoch):
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for batch in eval_data_loader:
            input, target = Variable(batch[0]), Variable(batch[1])
            if cuda:
                input = input.cuda(gpus_list[0])
                target = target.cuda(gpus_list[0])
              
            prediction = model(input)
                
            mse = criterionMSE(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            
        AVG_PSNR = avg_psnr / len(eval_data_loader)
        print("===> Epoch {} Complete: Avg. PSNR: {:.4f}".format(epoch, AVG_PSNR))
        
    return AVG_PSNR


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
    
    
if __name__ == '__main__':
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    
    print('===> Loading training datasets')
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
    
    print('===> Loading eval datasets')
    eval_set = get_eval_set(opt.data_dir, opt.hr_eval_dataset, opt.upscale_factor)
    eval_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=1, shuffle=True, pin_memory=True)
    
    
    print('===> Building model ', opt.model_type)
    if opt.model_type == 'UAN':
        model = UAN(num_channels=3, num_features=64, scale_factor=opt.upscale_factor) 
        
    model = torch.nn.DataParallel(model)
    criterionMSE = nn.MSELoss()
    criterionL1 = nn.L1Loss()
    criterionSL1 = nn.SmoothL1Loss()
    
    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')
    
    if opt.pretrained:
        model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
        if os.path.exists(model_name):
            model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')
    
    if cuda:
        model = model.cuda(gpus_list[0])
        criterionMSE = criterionMSE.cuda(gpus_list[0])
        criterionL1 = criterionL1.cuda(gpus_list[0])
        criterionSL1 = criterionSL1.cuda(gpus_list[0])
    
    tail_params = list(map(id, model.module.tail.parameters()))
    base_params = filter(lambda p: id(p) not in tail_params, model.parameters())
    
    optimizer = optim.Adam([{'params': base_params}, {'params': model.module.tail.parameters(), 'lr': opt.lr * 0.1},], lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
#    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    
    
    epoch_num, all_psnr = [], []
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        psnr = evaluate(epoch)
    
        if (epoch+1) % (500) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
                
        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(epoch)
            
        if opt.psnr_curve:
            all_psnr.append(psnr)
            epoch_num.append(epoch)
            plt.figure()
            plt.xlabel('epochs')
            plt.ylabel('PSNR')
            plt.plot(epoch_num, all_psnr)
            plt.show()
            
            psnr_data = np.array(all_psnr)
            np.save(opt.save_folder+'PSNR_curve.npy', psnr_data)