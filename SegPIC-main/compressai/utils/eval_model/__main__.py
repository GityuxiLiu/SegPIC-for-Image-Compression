# 12.12
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
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.zoo import load_state_dict, models

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = transforms.ToTensor()(Image.open(filepath).convert("RGB"))
    if args.crop:
        h, w = args.crop
    else:
        h, w = img.shape[-2:]
    # img = transforms.CenterCrop([h//64*64, w//64*64])(img)
    return img

def reconstruct(reconstruction, filename, recon_path):
    reconstruction = reconstruction.squeeze()
    reconstruction.clamp_(0, 1)
    reconstruction = transforms.ToPILImage()(reconstruction.cpu())
    reconstruction.save(os.path.join(recon_path, filename))

@torch.no_grad()
def inference(model, x, m, filename, recon_path):
    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()

    out_enc = model.compress(x_padded, args.grid)

    enc_time = time.time() - start
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], args.grid)
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    reconstruct(out_dec["x_hat"], filename, recon_path)         # add

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    info = {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

    bpp_allocate = {}
    for i in range(len(out_enc["strings"])):
        string = out_enc["strings"][i][0]
        bpp = len(string) * 8.0 / num_pixels
        bpp_allocate["bpp"+str(i)] = bpp
    info.update(bpp_allocate)
    
    return info

@torch.no_grad()
def inference_entropy_estimation(model, x, m, filename, recon_path):
    x = x.unsqueeze(0)
    start = time.time()

    out_net = model.forward(x, m, grid=args.grid)
    elapsed_time = time.time() - start

    if not os.path.exists(recon_path):
        os.makedirs(recon_path)
    reconstruct(out_net["x_hat"], filename, recon_path)

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    info = {
        "psnr": psnr(x, out_net["x_hat"]),
        "ms-ssim": ms_ssim(x, out_net["x_hat"], data_range=1.0).item(),
        "bpp": bpp.item(),
        "time": elapsed_time,  # broad estimation
    }
    return info

def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()

def eval_model(model, filepaths, entropy_estimation=False, half=False, recon_path='/opt/data/private/SAC/reconstruction', ifprint=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        _filename = f.split("/")[-1]
        x = read_image(f).to(device)
        if args.testNoMask:
            m = None
        else:
            img_name = os.path.basename(f)
            img_name = os.path.splitext(img_name)[0] + ".png"
            img_name = os.path.join(args.maskPath, img_name)
            m = read_image(img_name).to(device)

        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, m, _filename, recon_path)
        else:
            rv = inference_entropy_estimation(model, x, m, _filename, recon_path)
        for k, v in rv.items():
            metrics[k] += v
        if ifprint:
            print(rv)

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser()

    # Common options.
    parent_parser.add_argument("-d", "--dataset", type=str, help="dataset path")
    parent_parser.add_argument("-r", "--recon_path", type=str, default="reconstruction", help="where to save recon img")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    parent_parser.add_argument(
        "--maskPath",
        dest="maskPath",
        type=str,
        default=None,
        help="The mask path",
    )
    parent_parser.add_argument(
        "--testNoMask",
        action="store_true",
        help="use grid patitions as mask",
    )
    parent_parser.add_argument(
        "--grid",
        type=int,
        default=1,
        help="Grid patitions n x n",
    )
    parent_parser.add_argument(
        "--crop",
        type=int,
        nargs=2,
        default=None,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    return parent_parser

args = {}
def main(argv):
    parser = setup_args()
    global args
    args = parser.parse_args(argv)

    if args.testNoMask:
        print("test No mask")

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = args.paths
    opts = (args.architecture,)
    load_func = load_checkpoint
    log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        model.update(force=True)

        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.recon_path)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }
    print(args.paths)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
