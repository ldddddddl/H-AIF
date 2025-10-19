import torch.nn as nn
import torch
import numpy as np
from .model_utils import (
    deconv,
    predict_flow,
    conv3d,
    crop_like,
    init_weights,
    repeat_like,
    conv3d_lerelu_maxpl,
)
import sys
from dacite import from_dict
from dacite import Config as DaciteConfig
from omegaconf import OmegaConf

sys.path.append("..")
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig


class ImagesGenerate(nn.Module):
    def __init__(
        self,
        z_dim=128,
        initailize_weights=True,
        use_skip_layers=2,
        num_masks=8,
        batch_size=8,
        xlstm_cfg=None,
    ):
        super(ImagesGenerate, self).__init__()
        """
        Decodes the future frame and mask.解码光流和光流掩码
        """
        self.xlstm_cfg = xlstm_cfg
        self.z_dim = z_dim
        self.num_masks = num_masks
        self.batchsize = batch_size
        self.draw_feature = DrawFeature(xlstm_cfg=xlstm_cfg)
        self.cdna = CDNA(
            num_masks=num_masks, color_channels=3, kernel_size=[9, 9], z_dim=z_dim
        )
        self.flatten = nn.Flatten()
        self.optical_flow_conv = conv3d(
            xlstm_cfg.enc_out_dim // 2,
            128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.shape_error = conv3d(1, 1, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        ### 有时间整理下，消除魔法数字
        self.img_deconv6 = deconv(128, 128, kernel_size=(3, 3, 3))
        self.img_deconv5 = deconv(128, 64, stride=(1, 1, 1))
        # self.img_deconv4 = deconv(195, 64)
        # self.img_deconv3 = deconv(195, 64)
        self.img_deconv2 = deconv(193, 32, stride=(1, 2, 2))

        self.predict_optical_flow6 = predict_flow(128)
        # self.predict_optical_flow5 = predict_flow(195)
        # self.predict_optical_flow4 = predict_flow(195)
        self.predict_optical_flow3 = predict_flow(193)
        self.predict_optical_flow2 = predict_flow(99)
        self.predict_optical_press = predict_flow(
            12, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 1, 1)
        )

        self.upsampled_optical_flow6_to_5 = nn.ConvTranspose3d(
            in_channels=3,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        # self.upsampled_optical_flow5_to_4 = nn.ConvTranspose3d(
        #     xlstm_cfg.past_img_num, 1, (1, 3, 3), (1, 2, 2), bias=False
        # )
        self.upsampled_optical_flow4_to_3 = nn.ConvTranspose3d(
            1, 1, (1, 3, 3), (1, 1, 1), bias=False
        )
        self.upsampled_optical_flow3_to_2 = nn.ConvTranspose3d(
            1, 1, (1, 3, 3), (1, 2, 2), bias=False
        )
        self.upsample_conv = nn.ConvTranspose3d(
            65, 99, (1, 3, 3), (1, 2, 2), bias=False
        )

        self.predict_optical_flow2_masks = nn.Conv3d(
            in_channels=99,
            out_channels=9,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )
        self.predict_optical_flow2_mask = nn.Conv3d(
            in_channels=99,
            out_channels=1,
            kernel_size=(3,) * 3,
            stride=(1,) * 3,
            padding=(1,) * 3,
            bias=False,
        )

        self.softmax = nn.Softmax(dim=1)
        if initailize_weights:
            init_weights(self.modules())

    def forward(
        self,
        z_mix: torch.Tensor,
        last_frame: torch.Tensor,
        error: torch.Tensor,
        act: torch.Tensor,
    ):
        """
        Predicts the optical flow and optical flow mask.

        Args:
            z_mix: action conditioned z (output of fusion + action network)
            img_out_convs: outputs of the image encoders (skip connections)
        """
        B, S, C, H, W = error.shape
        act = act.reshape(
            B, 1, 1, self.xlstm_cfg.past_img_num + self.xlstm_cfg.future_img_num, -1
        )
        act_ = repeat_like(act, last_frame)
        z_mix_repeat = (
            z_mix.unsqueeze(1).unsqueeze(1).repeat(1, 1, C, 1, 1).transpose(-1, -2)
        )
        z_mix_repeat = repeat_like(crop_like(z_mix_repeat, last_frame), last_frame)
        error_out = self.shape_error(error)
        frame_with_error = torch.cat([last_frame, error_out, z_mix_repeat, act_], dim=1)
        frame_feat, frame_feat_results = self.draw_feature(frame_with_error)

        ###
        # error_feat, error_feat_results = self.draw_feature_error(error)
        use_skip_layers = len(frame_feat_results)
        out_img_conv1, out_img_conv2 = frame_feat_results  # out1 out3

        ### reshape
        if use_skip_layers != 0:
            optical_flow_in_f2 = self.optical_flow_conv(out_img_conv2)
            optical_flow_in_feat = self.img_deconv6(optical_flow_in_f2)
        if use_skip_layers == 2:
            # skip connection 1
            optical_flow4 = self.predict_optical_flow6(
                optical_flow_in_feat
            )  # torch.Size([8, 3, 4, 28, 28])
            optical_flow4_up = self.upsampled_optical_flow4_to_3(
                optical_flow4
            )  # torch.Size([8, 3, 8, 56, 56])
            out_img_deconv3 = self.img_deconv5(
                optical_flow_in_feat
            )  # torch.Size([8, 32, 8, 56, 56])

            out_img_conv2 = repeat_like(out_img_conv2, out_img_deconv3)
            frame_feat_ = repeat_like(frame_feat, out_img_deconv3)
            concat3 = torch.cat(
                (out_img_conv2, out_img_deconv3, optical_flow4_up, frame_feat_), dim=1
            )  # torch.Size([8, 163, 8, 56, 56])

            # skip connection 2
            optical_flow3 = self.predict_optical_flow3(
                concat3
            )  # torch.Size([8, 3, 8, 56, 56])
            optical_flow3_up = self.upsampled_optical_flow3_to_2(
                optical_flow3
            )  # torch.Size([8, 3, 8, 56, 56])
            out_img_deconv2 = self.img_deconv2(
                concat3
            )  # torch.Size([8, 32, 8, 56, 56])
            # press = self.predict_optical_press(out_img_conv1)
            out_img_conv1 = repeat_like(out_img_conv1, out_img_deconv2)
            concat2 = torch.cat(
                (out_img_conv1, out_img_deconv2, optical_flow3_up), dim=1
            )
            concat2 = self.upsample_conv(concat2)
        elif use_skip_layers == 0:
            out_deconv1 = self.deconv_out1(z_mix)
            out_deconv2 = self.deconv_out2(out_deconv1)
            out_deconv3 = self.deconv_out3(out_deconv2)
            # out_deconv4 = self.deconv_out4(out_deconv3)
            future_frame_unmasked = self.conv_pred(out_deconv3)
            future_frame_mask = self.conv_mask(out_deconv3)
            future_frame_masks = self.softmax(self.conv_masks(out_deconv3))

        if use_skip_layers != 0:
            future_frame_unmasked = self.predict_optical_flow2(
                concat2
            )  # torch.Size([8, 3, 8, 56, 56])
            future_frame_mask = self.predict_optical_flow2_mask(concat2)

            future_frame_masks = self.predict_optical_flow2_masks(concat2)
            future_frame_masks = self.softmax(future_frame_masks)

        future_frame = future_frame_unmasked * torch.sigmoid(future_frame_mask)

        if self.xlstm_cfg.is_use_cdna:
            # cdna_input = self.flatten(fused_result)
            cdna_input = self.flatten(out_img_conv2)

            # cdna transform
            next_transformed, cdna_kerns = self.cdna(last_frame, cdna_input)
            # skip connection
            # next_transformed = [_+future_frame for _ in next_transformed]
            masks = torch.split(future_frame_masks, 1, dim=1)
            masks = [crop_like(mask, last_frame) for mask in masks]
            output = masks[0] * last_frame

            for frame, mask in zip(next_transformed, masks[1:]):
                output += frame * mask
        else:
            output = future_frame

        return output


class ActGenerate(nn.Module):
    def __init__(
        self,
        input_dim=128,
        z_dim=128,
        device="cpu",
        xlstm_cfg=None,
        embed_dim=9,
        hidden_dim=128,
        num_layers=6,
        num_heads=9,
        output_dim=15,
    ):
        super().__init__()
        self.xlstm_cfg = xlstm_cfg
        self.modalities_seq_dim = xlstm_cfg.enc_out_dim * 2 + xlstm_cfg.act_model_enc.embedding_dim
        self.z_dim = z_dim
        self.device = device
        # self.linear1 = nn.Linear(xlstm_cfg.model.embedding_dim * 2, z_dim, bias=True, device=device)
        self.linear1 = nn.Linear(input_dim, z_dim, bias=True, device=device)
        # self.linear2 = nn.Linear(z_dim, z_dim//2, bias=True, device=device)
        self.linear3 = nn.Linear(z_dim, z_dim // 2, bias=True, device=device)
        # self.linear4 = nn.Linear(z_dim//4, z_dim//4, bias=True, device=device)
        self.linear5 = nn.Linear(560, 200, bias=True, device=device)
        if xlstm_cfg.data_format == "rpy":
            if xlstm_cfg.act_encoder == "xlstm":
                self.linear6 = nn.Linear(504, 3, bias=True, device=device)
            elif xlstm_cfg.act_encoder == "transformerxlstm":
                self.linear6 = nn.Linear(self.modalities_seq_dim, 3, bias=True, device=device)
        else:  # == joints
            # if xlstm_cfg.act_encoder == 'xlstm':
            self.linear6 = nn.Linear(
                self.modalities_seq_dim,
                xlstm_cfg.action_dim,
                bias=True,
                device=device,
            )
            # elif xlstm_cfg.act_encoder == 'transformerxlstm':
            #     self.linear6 = nn.Linear(1128, 9, bias=True, device=device)

        # self.std = nn.Linear(z_dim//4, 3, bias=True)
        # self.mean = nn.Linear(z_dim//4, 3, bias=True)
        self.lstm = nn.LSTM(z_dim // 2, z_dim // 2, 3, bias=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.relu_ = nn.ReLU()
        self.tanh_ = nn.Tanh()
        self.softplus = nn.Softplus()
        self.act_xlstm_layers = XLstmStack(self.xlstm_cfg.act_model_dnc, device=device)
        self.norm_ = nn.BatchNorm1d(xlstm_cfg.act_model_dnc.context_length)
        self.flatten = nn.Flatten(start_dim=2)
        self.conv1d = nn.Conv1d(xlstm_cfg.act_model_dnc.context_length, xlstm_cfg.horizon, kernel_size=1, padding=0)

    def forward(self, inp: torch.Tensor):
        """
        inp: z_mix
        tgt: action label

        """

        xlstm_out = self.act_xlstm_layers.xlstm(inp)
        xlstm_out = self.flatten(xlstm_out)
        xlstm_out = self.norm_(xlstm_out)
        xlstm_out = self.dropout(xlstm_out)
        # x = self.relu_(self.linear5(xlstm_out))
        # x = self.norm_(x)
        # x = self.dropout(x)
        # output, (h_n, c_n) = self.lstm(x)
        # x = self.linear4(x)
        x = self.conv1d(xlstm_out)
        action = self.tanh_(self.linear6(x))
        # action = action.clamp(-20.0, 240.0)
        # x = self.transformer_decoder(tgt, x)
        # mean = self.tanh_(self.mean(x)) * 2
        # std = self.softplus(self.std(x)) + 1e-5
        mean = None
        std = None
        # normal_dist = Normal(mean, std)
        # action = normal_dist.sample()
        # action = action.clamp(-2.0, 2.0)

        # x = self.linear6(x)
        # x = x.view(self.xlstm_cfg.batchsize, self.xlstm_cfg.future_img_num * self.xlstm_cfg.per_image_with_signal_num, 9)
        return mean, std, action


class SuckerAct(nn.Module):
    def __init__(self, z_dim=128, device="cpu", xlstm_cfg=None) -> None:
        super().__init__()
        self.xlstm_cfg = xlstm_cfg

        self.linear1 = nn.Linear(z_dim, z_dim, bias=True, device=device)
        self.out = nn.Linear(
            z_dim, 2, bias=True, device=device
        )
        self.norm = nn.BatchNorm1d(xlstm_cfg.horizon)
        self.dropout = nn.Dropout(0.2)
        self.relu_ = nn.ReLU()
        self.conv1d = nn.Conv1d(xlstm_cfg.act_model_dnc.context_length, xlstm_cfg.horizon, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.conv1d(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.relu_(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


class XLstmStack:
    def __init__(self, xlstm_cfg, device, shape:list | None=None) -> None:
        if shape is not None:
            xlstm_cfg.embedding_dim = np.prod(shape).item()
        
        self.xlstm_layer_config = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(xlstm_cfg),
            config=DaciteConfig(strict=True),
        )
        self.xlstm = xLSTMBlockStack(self.xlstm_layer_config)
        self.xlstm.reset_parameters()
        self.xlstm.to(device=device)


class DrawFeature(nn.Module):
    def __init__(
        self, in_channels=4, kernl_first_size=2, stride_first_size=1, xlstm_cfg=None
    ) -> None:
        super().__init__()
        self.xlstm_cfg = xlstm_cfg
        out_channels = self.xlstm_cfg.model.embedding_dim
        self.conv3d_1 = conv3d_lerelu_maxpl(
            in_channels, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.conv3d_2 = conv3d_lerelu_maxpl(
            32,
            32,
            kernel_size=(kernl_first_size, 3, 3),
            stride=(stride_first_size, 1, 1),
            padding=(0, 1, 1),
        )
        self.conv3d_3 = conv3d_lerelu_maxpl(
            32,
            out_channels,
            kernel_size=(kernl_first_size, 3, 3),
            stride=(stride_first_size, 1, 1),
            padding=(0, 1, 1),
        )

    def forward(self, x):
        if len(x.shape) == 4:
            # insert sequence dim
            x = x.unsqueeze(1)
        out_1 = self.conv3d_1(x)
        out_2 = self.conv3d_2(out_1)
        out_3 = self.conv3d_3(out_2)

        return out_3, [out_1, out_3]


RELU_SHIFT = 1e-10


class CDNA(nn.Module):
    def __init__(self, num_masks, color_channels, kernel_size, z_dim=128):
        super(CDNA, self).__init__()
        self.num_masks = num_masks
        self.color_channels = color_channels
        self.kernel_size = kernel_size  # 5*5
        self.cdna_params = nn.Linear(
            in_features=196608,
            out_features=self.kernel_size[0]
            * self.kernel_size[1]
            * self.color_channels,
        )

    def forward(self, prev_image, cdna_input):
        batch_size = cdna_input.shape[0]
        height = prev_image.shape[2]
        width = prev_image.shape[3]

        # Predict kernels using a linear function of the last hidden layer.
        cdna_kerns = self.cdna_params(cdna_input)

        # Reshape and normalize.
        cdna_kerns = cdna_kerns.view(
            batch_size, -1, self.kernel_size[0], self.kernel_size[1]
        )

        cdna_kerns = torch.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = torch.sum(cdna_kerns, (1, 2, 3), keepdim=True)
        cdna_kerns /= norm_factor

        # Treat the color channel dimension as the batch dimension since the same
        # transformation is applied to each color channel.
        # Treat the batch dimension as the channel dimension so that
        # depthwise_conv2d can apply a different transformation to each sample.
        # cdna_kerns = cdna_kerns.permute(0, 4, 1, 2, 3).squeeze(-1)#[8,5,5,1,10] --> [8,10,5,5]
        cdna_kerns = cdna_kerns.view(
            batch_size, self.color_channels, self.kernel_size[0], self.kernel_size[1]
        )  # [24, 3, 5, 5]
        # Swap the batch and channel dimensions.
        # prev_image = prev_image.permute(3, 1, 2, 0)
        # prev_image = prev_image.permute(1, 0, 2, 3) # [8, 3, 112, 112] --> [3, 112, 112, 8]

        # Transform image.
        # cdna_channel  -->  [out_channels, in_channels * groups, kernel_height, kernel_width]  [8, 3, 5, 5]
        # pre_image_channel --> [batch_size, in_channels, height, width]
        transformed = nn.functional.conv2d(
            prev_image.squeeze(1),
            cdna_kerns,
            padding=(self.kernel_size[0] - 1) // 2,
            groups=1,
            stride=1,
        )  # --> [channle, batch_size, h, w]

        # transformed = transformed.view(batch_size, self.color_channels, height, width, self.num_masks)
        # img_show(transformed[-1, :, :, :, -1])

        transformed = torch.split(transformed.unsqueeze(2), 1, dim=1)

        return transformed, cdna_kerns
