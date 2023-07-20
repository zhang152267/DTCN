import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Modeling.DerainDataset import *
from Modeling.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from Modeling.SSIM import SSIM
from Modeling.DTCN import *
from torchvision import datasets, transforms
import lpips
from loss_fun import *

parser = argparse.ArgumentParser(description="dtcn_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/DTCN/Model", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainH",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--use_contrast", type=bool, default=True,
                    help='use contrasive loss or not')
parser.add_argument("--use_lpis", type=bool, default=True,
                    help='use lpis loss or not')
opt = parser.parse_args()
loss_fn_vgg = lpips.LPIPS(net='alex').to(device) # choose between alexnet, VGG, or others
if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def LpisLoss(out_train, target_train, device):

    new_out_train = (torch.max(out_train)-out_train)/(torch.max(out_train)-torch.min(out_train))
    new_target_train = (torch.max(target_train)-target_train)/(torch.max(target_train)-torch.min(target_train))
    resize = transforms.Resize([256, 256])
    new_target_train = resize(new_target_train)
    new_out_train = resize(new_out_train)
    lpips_num = 0
    for ii in range(len(new_out_train)):
        outtrain = new_out_train[ii].reshape((1,3,256,256))
        targettrain = new_target_train[ii].reshape((1,3,256,256))
        lpips_num += float(loss_fn_vgg(targettrain.to(device), outtrain.to(device)))
        lpips_num = torch.tensor(lpips_num).to(device)

        return lpips_num

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = PReNet1(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    #criterion1 = nn.MSELoss(size_average=False)
    criterion = SSIM()
    loss_C = ContrastLoss().to(device)
    loss_ea = EdgeLoss().to(device)
    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()


            out_train, out_1, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            pixel_metric1 = - 5 * criterion(out_1, input_train - target_train)
            loss_eage1 = 0.025 * loss_ea(out_1, input_train - target_train)
            loss_ssim = - pixel_metric
            # Constrative loss
            loss_contrast = 4 * loss_C(out_train, target_train,
                                       input_train) if opt.use_contrast else 0  # scale the contrast loss
            loss_contrast1 = 0.5 * 4 * loss_C(out_1, input_train - target_train,
                                       input_train) if opt.use_contrast else 0  # scale the contrast loss
            # # LPIS loss
            loss_lpis = 10 * LpisLoss(out_train, target_train, device) if opt.use_lpis else 0  # scale the lpips loss
            loss_lpis1 = 0.5 * 10 * LpisLoss(out_1, input_train - target_train, device) if opt.use_lpis else 0  # scale the lpips loss
            loss_eage = 0.05 * loss_ea(out_train, target_train)

            loss = loss_ssim + 0.1 * pixel_metric1  + 0.1 * loss_contrast + 0.1* loss_contrast1 +loss_lpis + loss_lpis1 +loss_eage +loss_eage1

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _,_ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end

        # log the images
        model.eval()
        out_train, _, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')

    main()