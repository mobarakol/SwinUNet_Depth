import argparse
import os
import sys
import torch
from torch import nn
import torch.optim as optim
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from tqdm import tqdm
import numpy as np
from dataset import get_dataloader
from losses import custom_loss_function, RMSELoss, SSIM_Loss, SSIM_Loss_Lib, depth_loss
from torchmetrics import StructuralSimilarityIndexMeasure

def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--base_lr', type=float,  default=0.05,
                        help='segmentation network learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--cfg', type=str, metavar="FILE", 
                        default='configs/swin_tiny_patch4_window7_224_lite.yaml', help='path to config file', )
    parser.add_argument( "--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None, nargs='+',)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, ' 'full: cache all data, ' 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--ckpt', type=str, default='best_model.pth.tar', help='total gpu')
    parser.add_argument('--criterion', type=str, default='mse', help='total gpu')

    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

        
def train(model, trainloader, optimizer, criterion, iter_num, max_iterations, base_lr):
    model.train()
    for i_batch, s_batch in enumerate(trainloader):
        image_batch, label_batch = s_batch[0].cuda(), s_batch[1].cuda()
        #print(image_batch[0].min(), image_batch[0].max(),'min:', label_batch[0].min().item(),'max:', label_batch[0].max().item())
        #print(image_batch[0].min(), image_batch[0].max())
        outputs = model(image_batch)
        outputs = nn.Sigmoid()(outputs)
        loss = criterion(outputs, label_batch)
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

def test(model, testloader):
    model.eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=1)
    mse_all = []
    #rmse_all = []
    ssim_all = []
    with torch.no_grad():
        for i_batch, s_batch in enumerate(testloader):
            image_batch, label_batch = s_batch[0].cuda(), s_batch[1].cuda()
            outputs = model(image_batch)
            outputs = nn.Sigmoid()(outputs)
            mse = nn.MSELoss()(outputs, label_batch)
            mse_all.append(mse.item())
            #rmse_all.append(rmse(outputs, label_batch).item())
            ssim_all.append(ssim(outputs, label_batch).item())
        return np.mean(mse_all), np.mean(ssim_all)
            
def main(): 
    seed_everything()
    args = get_arg() 
    print('Running experiment on- criterion: {}, ckpt_dir:{}'.format(args.criterion, args.ckpt)) 
    trainloader, validloader = get_dataloader(args)
    config = get_config(args)
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)
    if args.n_gpu > 1:
            model = nn.DataParallel(model)

    if args.criterion == 'mse':        
        criterion = nn.MSELoss().cuda()
    elif args.criterion == 'depthloss':        
        criterion = depth_loss().cuda()
    elif args.criterion == 'l1':        
        criterion = nn.L1Loss().cuda()
    elif args.criterion == 'ssim':        
        #criterion = SSIM_Loss(data_range=1.0).cuda()
        criterion = SSIM_Loss_Lib(data_range=1.0).cuda()
    if args.criterion == 'rmse':        
        criterion = RMSELoss().cuda()
    elif args.criterion == 'bce':
        criterion = nn.BCELoss().cuda()
    elif args.criterion == 'custom': 
        criterion = custom_loss_function
    

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iterations = args.max_epochs * len(trainloader) 
    iter_num = 0
    best_mse, best_ssim, best_epoch = np.inf, 0, 0
    for epoch_num in range(args.max_epochs):
        train(model, trainloader, optimizer, criterion, iter_num, max_iterations, args.base_lr)
        mse, ssim = test(model, validloader)
        if mse < best_mse:
            best_mse = mse
            best_ssim = ssim
            best_epoch = epoch_num
            torch.save(model.state_dict(), args.ckpt)
            
        print('Epoch:{}, Curr MSE:{:.6f}, Best MSE:{:.6f}, Best SSIM:{:.6f}, Best Epoch:{}, Criterion:{}'.
                format(epoch_num, mse, best_mse, best_ssim, best_epoch, args.criterion))
        
        iter_num += 1

if __name__ == '__main__':
    main()
        
