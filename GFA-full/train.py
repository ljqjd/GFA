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
from data_loader import SYSUData, RegDBData, TestData, LLCMData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from utils import *
from loss.TripletLoss import OriTripletLoss, TripletLoss_WRT, OriTripletLoss1, orthonomal_loss, OrthonormalLoss
#from loss.ALT import AdaptiveTripletLoss
from loss.MSEL import MSEL, MSEL_Cos, MSEL_Feat
from loss.DiscriminativeCenterLoss import DCL
from tensorboardX import SummaryWriter
from model.make_model import build_vision_transformer
from config.config import cfg
from transformers import transform_rgb, transformaa, transform_thermal, transform_test

#import optimizer
#from scheduler import create_scheduler
#from optimizer import make_optimizer
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from model.select import Select
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.00035 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='adamw', type=str, help='optimizer')
parser.add_argument('--resume', '-r', default='bbbmarketlr_0.0003_time_20240930_213837_adamw_epoch_3.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='/home/jiaqi/GFA-full/save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=10, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='/home/jiaqi/GFA-full/log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='/home/jiaqi/GFA-full/log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=128, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int,
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
parser.add_argument('--seed', default=41, type=int,
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

elif dataset == 'llcm':
    data_path = '/mnt/data/ljq/LLCM/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [2, 1]  #[2, 1]: VIS to IR; [1, 2]: IR to VIS

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

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal, transform3=transformaa)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

net = build_vision_transformer(num_classes=n_class,cfg = cfg)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
    #    print(checkpoint.keys())
    #    start_epoch = checkpoint['epoch']
        start_epoch = 1
    #    net.load_state_dict(checkpoint['net'])
        net.load_state_dict(checkpoint['net'], strict=False)
    #    net.load_state_dict(checkpoint, strict=False)
       
  #      new_state_dict = {k: v for k, v in checkpoint.items() if 'classifier' not in k}
  #      net.load_state_dict(new_state_dict, strict=False)
    #    net.load_param(model_path)
      #  print('==> loaded checkpoint {} (epoch {})'
      #        .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss().to(device)
criterion_tri= OriTripletLoss(margin=args.margin).to(device)
criterion_tri1= OriTripletLoss1(margin=args.margin).to(device)
criterion_msel = MSEL(num_pos=args.num_pos, feat_norm='no')
criterion_dcl = DCL().to(device)

#criterion_alt = AdaptiveTripletLoss().to(device)
#criterion_orth = OrthonormalLoss().to(device)
select = Select().to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
elif args.optim == 'adamw':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)        
    

        

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 30:
        lr = args.lr
    elif epoch >= 30 and epoch < 45:
        lr = args.lr * 0.1
    elif epoch >= 45:
        lr = args.lr * 0.01
    elif epoch >= 80:
        lr = args.lr * 0.001    
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    tri_loss1 = AverageMeter()
    dcl_loss = AverageMeter()
    msel_loss = AverageMeter()
    orth_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
  #  scheduler.step(epoch)
    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, input3, label1, label2, label3) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)
        input1 = input1.to(device) 
        input2 = input2.to(device)
        input3 = input3.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)
        labels = labels.to(device)
        data_time.update(time.time() - end)
 
     
        scores, feats, orth = net(torch.cat([input1,input2,input3]))
        scores1, scores2, scores3, scores4, scores5 = scores.chunk(5,0)
        feats1, feats2, feats3, feats4, feats5 = feats.chunk(5,0)

        if epoch < 102 :
            feats44, label4 = select(feats1, feats4, scores1, scores4, label1)
            feats55, label5 = select(feats2, feats5, scores2, scores5, label2)
            loss_tri1, _ = criterion_tri(torch.cat([feats1,feats2]), torch.cat([label1,label2]))
            
            loss_id = criterion_id(scores2, label2) + criterion_id(scores1, label1) + criterion_id(scores3, label3) 
            loss_orth = orthonomal_loss(orth)*0.1
            loss_dcl = criterion_dcl(torch.cat([feats1,feats2]), torch.cat([label1,label2]))*0.5
            loss_msel = criterion_msel(torch.cat([feats1,feats2]), torch.cat([label1,label2]))*0.5
            loss_tri, batch_acc = criterion_tri(torch.cat([feats1,feats44,feats2,feats55]), torch.cat([label1,label4,label2,label5]))
            
            correct += (batch_acc / 2)
            _, predicted = torch.cat([feats1,feats2]).max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            loss = loss_id + loss_tri + loss_dcl + loss_msel + loss_orth + loss_tri1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            id_loss.update(loss_id.item())
            tri_loss.update(loss_tri.item())
            tri_loss1.update(loss_tri1.item())
            dcl_loss.update(loss_dcl.item())
            msel_loss.update(loss_msel.item())
            orth_loss.update(loss_orth.item())
            total += labels.size(0)
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
                      'TLoss1: {tri_loss1.val:.4f} ({tri_loss1.avg:.4f}) '
                      'oLoss: {orth_loss.val:.4f} ({orth_loss.avg:.4f}) '
                      'dclLoss: {dcl_loss.val:.4f} ({dcl_loss.avg:.4f}) '
                      'mselLoss: {msel_loss.val:.4f} ({msel_loss.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, orth_loss=orth_loss, dcl_loss=dcl_loss, msel_loss=msel_loss))
         
            writer.add_scalar('total_loss', train_loss.avg, epoch)
            writer.add_scalar('id_loss', id_loss.avg, epoch)
            writer.add_scalar('tri_loss', tri_loss.avg, epoch)
            writer.add_scalar('tri_loss1', tri_loss1.avg, epoch)
            writer.add_scalar('orth_loss', orth_loss.avg, epoch)
            writer.add_scalar('dcl_loss', dcl_loss.avg, epoch)
            writer.add_scalar('msel_loss', msel_loss.avg, epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            
            
            
            
        else :
        #    loss_idd = 0
        #    for i in range(210):
        #        loss_idd += criterion_id(score22[:,i,:], label2) + criterion_id(score11[:,i,:], label1)
        #    loss_idd = loss_idd * 0.5 / 210    
        #    loss_id = criterion_id(score2[:,0,:], label2) + criterion_id(score1[:,0,:], label1) + criterion_id(score3[:,0,:], label3) + loss_idd 
            loss_id = criterion_id(scores2, label2) + criterion_id(scores1, label1) + criterion_id(scores3, label3)  
            loss_dcl = criterion_dcl(torch.cat([feats1,feats2]), torch.cat([label1,label2]))*0.5
            loss_msel = criterion_msel(torch.cat([feats1,feats2]), torch.cat([label1,label2]))*0.5
            loss_tri, batch_acc = criterion_tri(torch.cat([feats1,feats2]), torch.cat([label1,label2]))
            correct += (batch_acc / 2)
            _, predicted = torch.cat([feats1,feats2]).max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            loss = loss_id + loss_tri + loss_dcl + loss_msel 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
            train_loss.update(loss.item())
            id_loss.update(loss_id.item())
       #     wtri_loss.update(loss_wtri.item())
            tri_loss.update(loss_tri.item())
            dcl_loss.update(loss_dcl.item())
            msel_loss.update(loss_msel.item())
      #      g_loss.update(loss_g.item())
            total += labels.size(0)
            torch.cuda.synchronize()

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'lr:{:.3f} '
                      'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
               #       'wTLoss: {wtri_loss.val:.4f} ({wtri_loss.avg:.4f}) '
                      'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
              #        'GLoss: {g_loss.val:.4f} ({g_loss.avg:.4f}) '
                      'dclLoss: {dcl_loss.val:.4f} ({dcl_loss.avg:.4f}) '
                      'mselLoss: {msel_loss.val:.4f} ({msel_loss.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, dcl_loss=dcl_loss, msel_loss=msel_loss))
         
            writer.add_scalar('total_loss', train_loss.avg, epoch)
            writer.add_scalar('id_loss', id_loss.avg, epoch)
      #      writer.add_scalar('wtri_loss', wtri_loss.avg, epoch)
            writer.add_scalar('tri_loss', tri_loss.avg, epoch)
        #    writer.add_scalar('g_loss', g_loss.avg, epoch)
            writer.add_scalar('dcl_loss', dcl_loss.avg, epoch)
            writer.add_scalar('msel_loss', msel_loss.avg, epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 768))
  
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
           
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 768))
  
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
           
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    
   distmat = re_ranking(query_feat1, gall_feat1)   

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
     
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
      
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)  
      
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
  
    return cmc, mAP, mINP



class DataLoaderVisualizer1:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def visualize(self, image_size=(128, 256)):
       
        plt.figure(figsize=(15, 5))

        for i, batch in enumerate(self.dataloader):
            images1, images2, images3, label1, label2, label3 = batch  

            unloader = transforms.ToPILImage()
            image1 = unloader(images1[0])
            image1 = image1.resize(image_size, Image.LANCZOS)
           
            plt.imshow(image1)
            plt.axis('off')
        #    for idx in range(len(images1)):  
              
         #       image1 = unloader(images1[idx])
         #       image1 = image1.resize(image_size, Image.LANCZOS)

         #       image2 = unloader(images2[idx])
         #       image2 = image2.resize(image_size, Image.LANCZOS)

         #       image3 = unloader(images3[idx])
         #       image3 = image3.resize(image_size, Image.LANCZOS)

         #       plt.subplot(4, 8, idx+1) 
         #       plt.imshow(image1)
         #       plt.axis('off')
         
        
            plt.show()  
            break  

class DataLoaderVisualizer:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def split_image_into_patches(self, image, patch_size=(16, 16), stride=12):
        """Split an image into patches."""
        w, h = image.size
        patch_width, patch_height = patch_size
        step_w, step_h = stride, stride

        patches = []
        for i in range(0, h - patch_height + 1, step_h):  # +1 to include the last patch
            for j in range(0, w - patch_width + 1, step_w):  # +1 to include the last patch
                patch = image.crop((j, i, j + patch_width, i + patch_height))
                patches.append(patch)
        return patches

    def visualize(self, image_size=(128, 256)):
        unloader = transforms.ToPILImage()
        plt.figure(figsize=(20, 20))  # Adjusted figure size for 20x10 grid

        for i, batch in enumerate(self.dataloader):
            images1, images2, images3, label1, label2, label3 = batch 

            # Choose the first image in the batch
            
            image1 = unloader(images2[0])
            
            image1 = image1.resize(image_size, Image.LANCZOS)

            # Split the image into patches
            patches = self.split_image_into_patches(image1)

            # Calculate the number of patches to display
            num_patches_to_display = 210  # 21 rows x 10 columns

            # Display the patches in a 20x10 grid
            for patch_idx, patch in enumerate(patches[:num_patches_to_display]):
                plt.subplot(21, 10, patch_idx + 1)
                plt.imshow(patch)
                plt.axis('off')

            # Adjust the layout to prevent overlap and reduce spacing between columns
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust wspace for column spacing
            plt.tight_layout()
            plt.show()
            break  # Break after the first batch for demonstration purposes

        



# training
print('==> Start Training...')
for epoch in range(start_epoch, 102 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    #print(epoch)
  #  print(trainset.cIndex)
  #  print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)
                     
                                  
#    visualizer = DataLoaderVisualizer1(trainloader)
#    visualizer.visualize()  
    
    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP = test(epoch)
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
      
        print('Best Epoch [{}]'.format(best_epoch))