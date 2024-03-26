import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from .utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.layers import conv1x1, conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel
import torch.nn.functional as F

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def interpolate(img, factor):
    with torch.no_grad():
        img = img.unsqueeze(1)
        img = F.interpolate(img.float(), scale_factor=factor, mode='nearest').long()
        img = img.squeeze(1)
    return img

class SegPIC(CompressionModel):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5

        enc_ch = [3, 192, 192, 192, 320]
        enc_groups = [1, 1, 1, 1]
        enc_pool_stride = [32, 16, 8, 4]
        enc_num_heads = [3, 6, 12, 24]

        dec_ch = [320, 192, 192, 192, 32]
        dec_groups = [1, 1, 1, 1]
        dec_num_heads = [24, 12, 6, 3]

        self.basic_E = nn.ModuleList(
            Downblock_SAL(
                enc_ch[i], enc_ch[i+1], kernel_size=5, stride=2, groups=enc_groups[i], 
                pool_stride=enc_pool_stride[i], num_heads=enc_num_heads[i]
            ) for i in range(4)
        )
        self.basic_D = nn.ModuleList(
            Upblock_SAL(
                dec_ch[i], dec_ch[i+1], kernel_size=5, stride=2, groups=dec_groups[i],
                  num_heads=dec_num_heads[i], inverse=False
            ) for i in range(4)
        )

        self.g_a1 = nn.Sequential(
            self.basic_E[0],
            self.basic_E[1],
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
        )
        self.g_a2 = nn.Sequential(
            self.basic_E[2],
            self.basic_E[3],
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s0 = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            self.basic_D[0],
            self.basic_D[1],
        )
        self.g_s1 = nn.Sequential(
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            self.basic_D[2],
            self.basic_D[3],
            conv1x1(32,3),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )
        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.g_ab = nn.Sequential(
            conv1x1(192, 128),
            GDN(128),
            conv1x1(128, 96),
            GDN(96),
            conv1x1(96, 96),
        )
        self.g_sb = nn.Sequential(
            conv1x1(96, 128),
            GDN(128, inverse=True),
            conv1x1(128, 192),
            GDN(192, inverse=True),
            conv1x1(192, 192),
        )
        self.g_ab2 = nn.Sequential(
            conv1x1(320, 192),
            GDN(192),
            conv1x1(192, 128),
            GDN(128),
            conv1x1(128, 96),
        )
        self.g_sb2 = nn.Sequential(
            conv1x1(96, 128),
            GDN(128, inverse=True),
            conv1x1(128, 192),
            GDN(192, inverse=True),
            conv1x1(192, 320),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.entropy_bottleneck_b = EntropyBottleneck(96)
        self.entropy_bottleneck_b2 = EntropyBottleneck(96)
        self.gaussian_conditional = GaussianConditional(None)

        self.dyn_encoder = RegionAdaTransform(in_ch=192, center_ch=192, kernel_size=3)
        self.dyn_decoder = RegionAdaTransform(in_ch=192, center_ch=192, kernel_size=3)
        self.dyn_enc_hyper = RegionAdaTransform(in_ch=320, center_ch=320)
        self.dyn_dec_hyper_means = RegionAdaTransform(in_ch=320, center_ch=320)
        self.dyn_dec_hyper_scales = RegionAdaTransform(in_ch=320, center_ch=320)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def bit_estimate(self, x, entropy_bottleneck):
        _,x_likelihoods = entropy_bottleneck(x)
        x_offset = entropy_bottleneck._get_medians()
        x_tmp = x - x_offset
        x_hat = ste_round(x_tmp) + x_offset
        return x_hat, x_likelihoods

    def mask_transmit(self, y, mask, g_ab, g_sb, entropy_bottleneck):
        list_center, _, list_mask = mask_pool2d(y, mask)
        mask_nums = [m.shape[0] for m in list_mask]
        centers = torch.cat(list_center, dim=0).unsqueeze(2).unsqueeze(3) #(b,192)

        z_y = g_ab(centers)
        z_y_hat, z_y_likelihoods = self.bit_estimate(z_y, entropy_bottleneck)
        y_recon_centers = g_sb(z_y_hat).squeeze()
        y_recon_centers = torch.split(y_recon_centers, mask_nums, dim=0)
        return z_y_hat, z_y_likelihoods, y_recon_centers, mask_nums, list_mask
    
    def grid_transmit(self, y, g_ab, g_sb, entropy_bottleneck, grid=4):
        centers = F.adaptive_avg_pool2d(y, (grid, grid))
        z_y = g_ab(centers)
        z_y_hat, z_y_likelihoods = self.bit_estimate(z_y, entropy_bottleneck)
        y_recon_centers = g_sb(z_y_hat)
        return z_y_hat, z_y_likelihoods, y_recon_centers
    
    def forward(self, x, mask=None, grid=4):
        if mask is not None:
            mask4 = interpolate(mask, 1/4)
            mask16 = interpolate(mask4, 1/4)

            y1 = self.g_a1(x)
            _, z_y1_likelihoods, y1_recon_centers, _, _ = self.mask_transmit(y1, mask4, self.g_ab, self.g_sb, self.entropy_bottleneck_b)
            y1 = self.dyn_encoder(y1, y1_recon_centers, mask4)
            y = self.g_a2(y1)
            _, z_y2_likelihoods, y2_recon_centers, _, _ = self.mask_transmit(y, mask16, self.g_ab2, self.g_sb2, self.entropy_bottleneck_b2)
            y_dyn_mask = self.dyn_enc_hyper(y, y2_recon_centers, mask16)
        else:
            y1 = self.g_a1(x)
            _, z_y1_likelihoods, y1_recon_centers = self.grid_transmit(y1, self.g_ab, self.g_sb, self.entropy_bottleneck_b, grid=grid)
            y1 = self.dyn_encoder(y1, y1_recon_centers)
            y = self.g_a2(y1)
            _, z_y2_likelihoods, y2_recon_centers = self.grid_transmit(y, self.g_ab2, self.g_sb2, self.entropy_bottleneck_b2, grid=grid)
            y_dyn_mask = self.dyn_enc_hyper(y, y2_recon_centers)

        z = self.h_a(y_dyn_mask)
        z_hat, z_likelihoods = self.bit_estimate(z, self.entropy_bottleneck)
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        if mask is not None:
            latent_scales = self.dyn_dec_hyper_scales(latent_scales, y2_recon_centers, mask16)
            latent_means = self.dyn_dec_hyper_means(latent_means, y2_recon_centers, mask16)
        else:
            latent_scales = self.dyn_dec_hyper_scales(latent_scales, y2_recon_centers)
            latent_means = self.dyn_dec_hyper_means(latent_means, y2_recon_centers)

        y_shape = y.shape[2:]
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        
        y_hat = self.g_s0(y_hat)
        if mask is not None:
            y_hat = self.dyn_decoder(y_hat, y1_recon_centers, mask4)
        else:
            y_hat = self.dyn_decoder(y_hat, y1_recon_centers)
        x_hat = self.g_s1(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "z_center": z_y1_likelihoods,
                             "z_center2": z_y2_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)
    
    def grid_compress(self, y, g_ab, g_sb, entropy_bottleneck, grid=None):
        centers = F.adaptive_avg_pool2d(y, (grid, grid))
        z_y = g_ab(centers)
        shape = z_y.shape[-2:]
        z_y_string = entropy_bottleneck.compress(z_y)
        z_y_hat = entropy_bottleneck.decompress(z_y_string, shape)
        y_recon_centers = g_sb(z_y_hat)
        return z_y_hat, z_y_string, y_recon_centers
    
    # def mask_compress(self, y, mask, g_ab, g_sb, entropy_bottleneck):
    #     list_center, _, list_mask = mask_pool2d(y, mask)
    #     mask_nums = [m.shape[0] for m in list_mask]
    #     centers = torch.cat(list_center, dim=0).unsqueeze(2).unsqueeze(3) #(b,192)

    #     z_y = g_ab(centers)
    #     shape = z_y.shape[-2:]
    #     z_y_string = entropy_bottleneck.compress(z_y)
    #     z_y_hat = entropy_bottleneck.decompress(z_y_string, shape)
    #     y_recon_centers = g_sb(z_y_hat).squeeze()
    #     y_recon_centers = torch.split(y_recon_centers, mask_nums, dim=0)
    #     return z_y_hat, z_y_likelihoods, y_recon_centers, mask_nums, list_mask

    def compress(self, x, grid=4):
        y1 = self.g_a1(x)
        _, z_y1_string, y1_recon_centers = self.grid_compress(y1, self.g_ab, self.g_sb, self.entropy_bottleneck_b, grid)
        y1 = self.dyn_encoder(y1, y1_recon_centers)
        y = self.g_a2(y1)

        _, z_y2_string, y2_recon_centers = self.grid_compress(y, self.g_ab2, self.g_sb2, self.entropy_bottleneck_b2, grid)
        y_dyn_mask = self.dyn_enc_hyper(y, y2_recon_centers)

        y_shape = y.shape[2:]

        z = self.h_a(y_dyn_mask)
        # do
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        latent_scales = self.dyn_dec_hyper_scales(latent_scales, y2_recon_centers)
        latent_means = self.dyn_dec_hyper_means(latent_means, y2_recon_centers)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings, z_y1_string, z_y2_string], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape, grid=4):
        z_y1_string, z_y2_string = strings[2], strings[3]
        z_y1_shape = [grid, grid]
        z_y2_shape = [grid, grid]
        z_y1_hat = self.entropy_bottleneck_b.decompress(z_y1_string, z_y1_shape)
        z_y2_hat = self.entropy_bottleneck_b2.decompress(z_y2_string, z_y2_shape)
        y2_recon_centers = self.g_sb2(z_y2_hat)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        latent_scales = self.dyn_dec_hyper_scales(latent_scales, y2_recon_centers)
        latent_means = self.dyn_dec_hyper_means(latent_means, y2_recon_centers)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_hat = self.g_s0(y_hat)
        y1_recon_centers = self.g_sb(z_y1_hat)
        y_hat = self.dyn_decoder(y_hat, y1_recon_centers)
        x_hat = self.g_s1(y_hat)

        return {"x_hat": x_hat}

# ===================================================================================================
# ===================================================================================================
# ===================================       Bound       =============================================
# ===================================================================================================
# ===================================================================================================
nums = 0
sum_enc = 0
sum_dec = 0

class RegionAdaTransform(nn.Module):
    def __init__(self, in_ch=192, center_ch=192, mid_ch=None, kernel_size=3):
        super().__init__()
        self.ch = in_ch
        if mid_ch is None:
            mid_ch = in_ch
        self.mid_ch = mid_ch
        self.k = kernel_size

        self.conv = nn.Sequential(
            conv1x1(in_ch, mid_ch),
            nn.LeakyReLU(negative_slope=0.1),
            conv1x1(mid_ch, mid_ch),
        )
        self.k_sp_gen = nn.Sequential(
            conv1x1(mid_ch+center_ch, mid_ch),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_ch, mid_ch, stride=1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_ch, mid_ch*kernel_size**2, stride=1, kernel_size=kernel_size, 
                      padding=(kernel_size-1)//2, groups=mid_ch),
        )
        self.k_ch_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv1x1(mid_ch, mid_ch),
            nn.LeakyReLU(negative_slope=0.1),
            conv1x1(mid_ch, mid_ch),
            nn.Sigmoid(),
        )
        self.conv_out = conv1x1(mid_ch, in_ch)

    def forward(self, input, list_center, batch_mask=None):
        """
        input:
            input          (bs, channel, h, w)
            list_center    bs,(mask_num, ch)
            batch_mask     (bs, w, h)
        output:
            out            (bs, ch_out, w, h)

        """
        bs, _, h, w = input.shape
        k = self.k
        ch = self.mid_ch
        if batch_mask is not None:
            centers_exp = mask_expand(list_center, batch_mask)
        else:
            centers_exp = F.interpolate(list_center, size=(h,w), mode='nearest')

        input_shortcut = input
        input = self.conv(input)
        k_sp = self.k_sp_gen(torch.cat([input, centers_exp], dim=1)).reshape(bs, ch, k**2,h,w)
        out = self.dynamic_conv(input, k_sp, k)
        out = self.k_ch_gen(out)*out
        out = self.conv_out(out)
        return  out + input_shortcut

    def dynamic_conv(self, input, k_sp, k):
        bs, ch, h, w = input.shape
        unfold = torch.nn.Unfold(kernel_size=(k,k), padding=(k-1)//2)
        input = unfold(input).view(bs, ch, k**2, h, w)
        return torch.einsum('bckhw,bckhw->bchw', input, k_sp)

def mask_pool2d(batch_feat, batch_mask):
    """
    input:
        batch_feat  (bs, ch, w, h)
        batch_mask  (bs, w, h)
    output:
        list_feat_center    bs,(mask_num, ch)
        bacth_feat_exp      (bs, ch, w, h)
        list_mask           bs,(mask_num, w, h)
    """
    list_feat_exp = []
    list_feat_center = []
    list_mask = []
    bs = batch_feat.size()[0]
    
    for i in range(bs):
        feat = batch_feat[i]
        mask = batch_mask[i]

        with torch.no_grad():
            mask_oh = F.one_hot(mask)
            mask_pm = mask_oh.permute(2,0,1)
            mask_usq = mask_pm.unsqueeze(0)
            area = torch.sum(mask_usq, dim=(-2,-1)) + 0.0005
        feat_usq = feat.unsqueeze(1)

        feat_center = torch.sum(feat_usq*mask_usq, dim=(-2,-1))/area
        expand_feat = feat_center[:, mask]

        list_feat_center.append(feat_center.permute(1,0))
        list_feat_exp.append(expand_feat)  
        list_mask.append(mask_pm)
    bacth_feat_exp = torch.stack(list_feat_exp)
    return list_feat_center, bacth_feat_exp, list_mask

def mask_pool2d_sem(batch_feat, batch_mask, batch_cats, len_cats=16, class_cats=133, isoutexp = False):
    """
    input:
        batch_feat  (bs, ch, w, h)
        batch_mask  (bs, w, h)
        batch_cats  (bs, 16)
    output:
        list_feat_center    bs,(mask_num, ch)
        list_feat_exp      (bs, ch, w, h)
        list_mask           bs,(mask_num, w, h)
        list_cats           bs,(mask_num, class_cats)
    """
    list_feat_exp = []
    list_feat_center = []
    list_mask = []
    list_cats = []
    bs = batch_feat.size()[0]
    for i in range(bs):
        feat = batch_feat[i]
        mask = batch_mask[i]
        cats = batch_cats[i]
        
        with torch.no_grad():
            ids, areas = torch.unique(mask, return_counts=True)
            cats = torch.Tensor([cats[x] for x in ids]).cuda().long()
            centers_sem = F.one_hot(cats, class_cats).float() # (mask_num,class_cats)
            list_cats.append(centers_sem)

            mask_pm = []
            for id in ids:
                m = torch.zeros_like(mask).bool() #(w,h)
                m[mask==id] = 1
                mask_pm.append(m)
            mask_pm = torch.stack(mask_pm) #(mask_num,w,h)
            mask_usq = mask_pm.unsqueeze(0) #(1,mask_num,w,h)
            assert mask_pm.shape[0] == centers_sem.shape[0], f"len_mask_pm={mask_pm.shape[0]}, len_sem={centers_sem.shape[0]}"

        feat_usq = feat.unsqueeze(1) #(c,1,w,h)
        feat_center = torch.sum(feat_usq*mask_usq, dim=(-2,-1))/areas #(c,mask_num)/(mask_num)

        if isoutexp:
            list_feat_exp.append(feat_center[:, mask])  
        list_feat_center.append(feat_center.permute(1,0))
        list_mask.append(mask_pm)
    if isoutexp:
        list_feat_exp = torch.stack(list_feat_exp)
    return list_feat_center, list_feat_exp, list_mask, list_cats

def mask_expand(list_center, batch_mask):
    """
    input:
        list_center  bs,(mask_num, ch)
        batch_mask  (bs, w, h)
    output:
        bacth_feat_exp      (bs, ch, w, h)
    """
    list_feat_exp = []
    bs = batch_mask.size()[0]
    
    for i in range(bs):
        mask = batch_mask[i]
        feat_center = list_center[i]
        expand_feat = feat_center[mask, :].permute(2,0,1) #(w,h,ch)
        list_feat_exp.append(expand_feat)  
    return torch.stack(list_feat_exp).contiguous()

class Downblock_SAL(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, groups, pool_stride, num_heads):
        super().__init__()
        self.conv_groups = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups
            ),
            GDN(out_c),
        )
        self.scales_bias = nn.Sequential(
            conv1x1(out_c, out_c),
            nn.GELU(),
            conv1x1(out_c, out_c*2),
        )
    
    def forward(self, x):
        x = self.conv_groups(x)        
        x_q = self.scales_bias(x)
        ch = x_q.shape[1]
        x_scales = x_q[:,:ch//2,:,:]
        x_bias = x_q[:,ch//2:,:,:]
        x = x * (1 + x_scales) + x_bias
        return x

class Upblock_SAL(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, groups, num_heads, inverse=False):
        super().__init__()
        self.conv_groups = nn.Sequential(
            nn.ConvTranspose2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                output_padding=stride - 1,
                padding=kernel_size // 2,
                groups=groups,
            ),
            GDN(out_c, inverse=inverse),
        )
        self.scales_bias = nn.Sequential(
            conv1x1(in_c, in_c),
            nn.GELU(),
            conv1x1(in_c, in_c*2),
        )

    def forward(self, x):
        x_q = self.scales_bias(x)
        ch = x_q.shape[1]
        x_scales = x_q[:,:ch//2,:,:]
        x_bias = x_q[:,ch//2:,:,:]
        x = x * (1 + x_scales) + x_bias
        x = self.conv_groups(x)
        return x
