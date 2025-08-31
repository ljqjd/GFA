from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import *
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from utils import *
from loss.TripletLoss import OriTripletLoss, TripletLoss_WRT, OriTripletLoss1
#from loss.CenterTripletLoss import CenterTripletLoss
#from loss.DiscriminativeCenterLoss import DCL
from loss.CenterLoss import CenterLoss
from tensorboardX import SummaryWriter
from model.make_model import build_vision_transformer
from config.config import cfg,_C
from transformers import transform_rgb, transformaa, transform_thermal, transform_test
from new.make_dataloader import make_dataloader
from new.metrics import R1_mAP_eval


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='msmt17', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.0004 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='adamw', type=str, help='optimizer')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='/home/jiaqi/Baseline2pre/save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=10, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='/home/jiaqi/Baseline2pre/log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='/home/jiaqi/Baseline2pre/log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.1, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/mnt/data/ljq/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'

elif dataset == 'regdb':
    data_path = '/mnt/data/ljq/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    
elif dataset == 'msmt17':
    data_path = '/mnt/backup/qingjie/data/MSMT17_V1/'
    log_path = args.log_path + 'msmt17_log/'   
     
elif dataset == 'market':
    data_path = '/mnt/backup/qingjie/market1501/'
    log_path = args.log_path + 'market_log/'  

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
suffix = dataset
suffix = suffix + 'lr_{}_time_{}'.format(args.lr, cur_time)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code



end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform1=transform_rgb, transform2=transform_thermal, transform3=transformaa)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal, transform3=transformaa)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')


print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

trainloader, train_loader_normal, val_loader, num_query, n_class, camera_num, view_num = make_dataloader(_C)

net = build_vision_transformer(num_classes=n_class,cfg = cfg)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
       # start_epoch = checkpoint['epoch']
        start_epoch = 1
        net.load_state_dict(checkpoint['net'], strict=False)
      #  print('==> loaded checkpoint {} (epoch {})'
      #        .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss().to(device)
#criterion_tri = TripletLoss_WRT()
criterion_tri= OriTripletLoss(margin=args.margin).to(device)
criterion_tri1= OriTripletLoss1(margin=args.margin).to(device)
#criterion_cen= CenterLoss(num_classes=n_class).to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) + list(map(id, net.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
elif args.optim == 'adamw':
    ignored_params = list(map(id, net.bottleneck.parameters())) 
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr}
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)        
    

        

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 30:
        lr = args.lr * 0.1
    elif epoch >= 30:
        lr = args.lr * 0.01
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
  #  cen_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, input3, label1) in enumerate(trainloader):     #input2:id,input3:cam id
        input1 = input1.to(device) 
        input2 = torch.tensor(input2).to(device)
        data_time.update(time.time() - end)
       # print(input2)
        scores, feats = net(input1)
         
    #    scores, score1, score2, score3, score4, score5, feats, feat1, feat2, feat3, feat4, feat5 = net(torch.cat([input1]))
        
        
     #   loss_idd1 = criterion_id(score1, input2) 
     #   loss_idd2 = criterion_id(score2, input2) 
     #   loss_idd3 = criterion_id(score3, input2) 
     #   loss_idd4 = criterion_id(score4, input2) 
     #   loss_idd5 = criterion_id(score5, input2) 
     #   loss_tri1 = criterion_tri1(feat1, input2)
     #   loss_tri2 = criterion_tri1(feat2, input2)
     #   loss_tri3 = criterion_tri1(feat3, input2)
     #   loss_tri4 = criterion_tri1(feat4, input2)
     #   loss_tri5 = criterion_tri1(feat5, input2)
          
     #   loss_idd = loss_idd1 + loss_idd2 + loss_idd3 + loss_idd4 + loss_idd5 + loss_tri1 + loss_tri2 + loss_tri3 + loss_tri4 + loss_tri5
     #   loss_idd = loss_idd * 0.5 / 5   
          
     #   loss_id = criterion_id(scores, input2) + loss_idd 
        
        loss_id = criterion_id(scores, input2)
     #   loss_cen = criterion_cen(feats, input2)
        loss_tri, batch_acc = criterion_tri(feats, input2)
        correct += (batch_acc / 2)
        _, predicted = feats.max(1)
        correct += (predicted.eq(input2).sum().item() / 2)


        loss =  loss_tri + loss_id 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
    #    cen_loss.update(loss_cen.item(), 2 * input1.size(0))

        total += input2.size(0)
        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
          #        'CLoss: {cen_loss.val:.4f} ({cen_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))
         
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
 #   writer.add_scalar('cen_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    


def test(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    evaluator.reset()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = net(img)
            evaluator.update((feat, vid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
 #   logger.info("Validation Results - Epoch: {}".format(epoch))
 #   logger.info("mAP: {:.1%}".format(mAP))
 #   for r in [1, 5, 10]:
 #       logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    torch.cuda.empty_cache()
 #   writer.add_scalar('rank1', cmc[0], epoch)
 #   writer.add_scalar('mAP', mAP, epoch)
 #   writer.add_scalar('mINP', mINP, epoch)

    return cmc, mAP




# training
print('==> Start Training...')
for epoch in range(start_epoch+1, 101 - start_epoch):

    print('==> Preparing Data Loader...')
    train(epoch)

    if epoch > 0 and epoch % 10 == 0:
 
        # save model
    
            state = {
                'net': net.state_dict(),
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

