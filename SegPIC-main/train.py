# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from compressai.datasets import ImageFolder, ImageFolder_nomask
from compressai.zoo import models
from pytorch_msssim import ms_ssim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["msssim_loss"] = 1-ms_ssim(target, output["x_hat"], data_range=1.0)
        out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )

        if args.lossMode == "ms_ssim":
            out["loss"] = self.lmbda * out["msssim_loss"] + out["bpp_loss"]
        else:
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="segpic",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "--useMask", action="store_true", default=False, help="Use Mask during training"
    )
    parser.add_argument(
        "--lrReset", action="store_true", default=False, help="Reset LR schedule"
    )
    parser.add_argument(
        "--lossMode",
        default="psnr",
        help="psnr or ms_ssim",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=8,
        help="LR reduce patience",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="LR reduce eps",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="LR factor",
    )
    parser.add_argument(
        "--p_aug",
        type=float,
        default=0,
        help="Probability of data augment",
    )
    parser.add_argument(
        "--saveStep",
        type=int,
        default=10,
        help="save model each n epochs",
    )
    parser.add_argument(
        "--newLambda", action="store_true", default=False, help="If fine-tune a new Lambda"
    )
    parser.add_argument(
        "--data_num",
        type=int,
        default=10000000,
        help="Data num one epoch",
    )
    args = parser.parse_args(argv)
    return args

tb_logger = SummaryWriter('./events')
def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    print("\n",datetime.datetime.now())
    model.train()
    device = next(model.parameters()).device
    
    t0 = datetime.datetime.now()
    
    for i, d in enumerate(train_dataloader):

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        if type(d) is list:
            d = [x.to(device) for x in d]
            out_net = model(*d)
            d = d[0]
        else:
            d = d.to(device)
            out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            dt = deltatime.seconds + 1e-6 * deltatime.microseconds

            loss = out_criterion["loss"].item()
            mse_loss = out_criterion["mse_loss"].item() * 255 ** 2 / 3

            msssim_loss = out_criterion["msssim_loss"].item()
            bpp_loss = out_criterion["bpp_loss"].item()
            aux_loss = aux_loss.item()
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss:.3f} |'
                f'\tTime: {dt:.2f} |'
                f'\tMSE loss: {mse_loss:.3f} |'
                f'\tMS-SSIM loss: {-10*math.log10(msssim_loss):.3f} |'
                f'\tBpp loss: {bpp_loss:.2f} |'
                f'\tAux loss: {aux_loss:.2f} |'
            )
            global_step = i + epoch*len(train_dataloader.dataset)/len(d)
            tb_logger.add_scalar('loss', loss, global_step)
            tb_logger.add_scalar('mse_loss', mse_loss, global_step)
            tb_logger.add_scalar('bpp_loss', bpp_loss, global_step)
            tb_logger.add_scalar('aux_loss', aux_loss, global_step)
            t0 = t1
            if i>args.data_num:
                break

def test_epoch(epoch, test_dataloader, model, criterion):
    
    print(datetime.datetime.now())
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            if type(d) is list:
                d = [x.to(device) for x in d]
                out_net = model(*d)
                d = d[0]
            else:
                d = d.to(device)
                out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
    
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg.item():.3f} |"
        f"\tMSE loss: {mse_loss.avg.item() * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg.item():.2f} |"
        f"\tAux loss: {aux_loss.avg.item():.2f}\n"
    )
    return loss.avg

args = {}
def main(argv):
    print("\n",datetime.datetime.now())
    global args
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size)]
    )
    
    if args.useMask:
        print("Use Mask")
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, p_aug=args.p_aug)
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, p_aug=args.p_aug)
    else:
        print("No Mask")
        train_dataset = ImageFolder_nomask(args.dataset, split="train", transform=train_transforms, p_aug=args.p_aug)
        test_dataset = ImageFolder_nomask(args.dataset, split="test", transform=test_transforms, p_aug=args.p_aug)
    print("Probability of Data Augment: ", args.p_aug)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model]()
    net = net.to(device)

    # total = sum([param.nelement() for param in net.parameters()])
    # print("Number of parameter: %.4fM" % (total/1e6))
    
    if args.lossMode == "ms_ssim":
        print("loss_mode: ms_ssim")
    else:
        print("loss_mode: psnr")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    
    best_loss = float("inf")
    last_epoch = 0

    if args.checkpoint:  # load from previous checkpoint
        print("Loading:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"] + 1
        else:
            last_epoch = 0
        
        if args.cuda and torch.cuda.device_count() > 1:
            weights_dict = {}
            state_dict = checkpoint["state_dict"]
            for k, v in state_dict.items():
                if "module." not in k:
                    k = "module." + k
                weights_dict[k] = v
            net.load_state_dict(weights_dict)
        else:
            weights_dict = {}
            state_dict = checkpoint["state_dict"]
            for k, v in state_dict.items():
                if "module." in k:
                    k = k.replace('module.','')
                weights_dict[k] = v
            net.load_state_dict(weights_dict)
        
        if not args.lrReset and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"]) 
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "loss" in checkpoint and not args.newLambda:
            best_loss = checkpoint["loss"]
            print("best_loss: ", best_loss.item())
        del checkpoint
        torch.cuda.empty_cache()
        
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        ckp_state = {
            "epoch": epoch,
            "loss": loss,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        # ckp_state2 = {
        #     "state_dict": net.state_dict(),
        # }

        if args.save:
            save_checkpoint(ckp_state, is_best, args.save_path,)
            # save_checkpoint(ckp_state2, is_best, args.save_path,)
        
        if args.saveStep > 0:
            if epoch % args.saveStep == 0:
                filename = args.save_path[:-8]+"_e"+ str(epoch) + args.save_path[-8:]
                save_checkpoint(
                    state=ckp_state,
                    is_best=False,
                    filename=filename
                )
                print("save this epoch: ", filename)
        print()

if __name__ == "__main__":
    main(sys.argv[1:])


