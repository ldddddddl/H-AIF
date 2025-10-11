import torch.nn as nn
import torch.nn.functional as F
import torch
# from model_utils import CausalConv1D
import sys
# sys.path.append(r"/home/ubuntu/Desktop/action_generation/model")
from .model_utils import CausalConv2D, conv3d_lerelu_maxpl, gaussian_parameters, sample_gaussian, conv3d, DiffDecoder, soft_pool2d, soft_pool3d
from .pusht_decoders import ActGenerate, ImagesGenerate, XLstmStack
from .convlstm import ConvLSTM
from .pusht_critic import Critic
from misc import VariableContainer
sys.path.append("..")

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from model.snn import CSNN, SNNActDecoder, DownConvSNN

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

class ActNet(nn.Module):
    def __init__(self, xlstm_cfg, is_use_cuda=False, device='vanilla'):
        super(ActNet, self).__init__()
        self.device = device
        self.xlstm_cfg = xlstm_cfg
        # self.enc_out_dim = (xlstm_cfg.past_img_num * xlstm_cfg.future_img_num) * 2 + xlstm_cfg.per_image_with_signal_num
        self.enc_out_dim = xlstm_cfg.model.embedding_dim * 2

        self.compute_error = ComputeLIFResiduals(is_use_snn_residual=xlstm_cfg.is_use_snn_residual)
        self.modal_fusion_model = MultiModalFusionModel(num_modalities=2, z_dim=self.xlstm_cfg.z_dim, input_dim=self.enc_out_dim, xlstm_cfg=xlstm_cfg)
        # self.modal_fusion_model.apply(init_weights)
        if xlstm_cfg.act_encoder == 'causalconv':
            self.causal_model = CausalNet(z_dim=self.enc_out_dim, xlstm_cfg=xlstm_cfg)
        elif xlstm_cfg.act_encoder == 'transformerxlstm':
            self.transbilstm = TransformerXLSTM(input_channels=4, embed_dim=self.enc_out_dim * 2, hidden_dim=self.enc_out_dim * 2, num_layers=xlstm_cfg.transformer.num_layers, num_heads=xlstm_cfg.transformer.num_heads, output_dim=60, max_history=5, config=xlstm_cfg)
        elif xlstm_cfg.act_encoder == 'xlstm': 
            xlstm_cfg.act_model_enc.num_blocks = xlstm_cfg.transformer.num_layers + xlstm_cfg.act_model_enc.num_blocks
            self.act_xlstm = XLstmStack(self.xlstm_cfg.act_model_enc, device=device)
        else:
            raise ValueError('action encoder keyword error')
        
        if self.xlstm_cfg.visual_encoder == 'diffusion':
            self.con3d_1 = conv3d_lerelu_maxpl(64, 64, (3, 3, 3), (1, 1, 1), (1, 2, 2))
            self.con3d_2 = conv3d_lerelu_maxpl(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            # 
            
            self.obs_diff = ImageDiffusion()

            self.obs_dnc = DiffDecoder(xlstm_cfg, 1, 1)
            self.conv_lstm = ConvLstmEnc(input_channels=3, output_channel=128)
        else:
            self.obs_xlstm_layers = XLstmStack(self.xlstm_cfg.model, device=self.device)
            self.obs_img_generate = ImagesGenerate(z_dim=self.xlstm_cfg.z_dim, batch_size=self.xlstm_cfg.batchsize, xlstm_cfg=xlstm_cfg)

            

        if self.xlstm_cfg.is_diff_generate_act:
            unet1d = Unet1D(dim = 8, dim_mults = (1, 2, 4, 8), channels = 100)
            self.diffusion = GaussianDiffusion1D(
                unet1d,
                seq_length = 384,
                timesteps = 500,
                objective = 'pred_v'
            )
            
        if xlstm_cfg.snn.is_use:
            self.act_generate = SNNActDecoder(act_dim=9, hidden_dim=120, config=xlstm_cfg)
        else:
            self.act_generate = ActGenerate(input_dim=self.enc_out_dim, z_dim=self.xlstm_cfg.z_dim, device=self.device, xlstm_cfg=xlstm_cfg)
        # self.sucker_act = SuckerAct(z_dim=self.xlstm_cfg.z_dim, device=self.device, xlstm_cfg=xlstm_cfg)
        # print(self.xlstm_layers.xlstm_layer_config)
        self.output = VariableContainer()
        self.self_attentions = SelfAttentions(seq_len=self.enc_out_dim)
        self.act_attn = SelfAttentions(input_dim=188, seq_len=self.enc_out_dim, eq_output=True)
        # self.self_attentions.apply(init_weights)
        # print(self.xlstm_layers.xlstm)
        self.in_channels = self.xlstm_cfg.past_img_num + self.xlstm_cfg.future_img_num

        self.obs_init_conv3d = conv3d(self.in_channels, 
                                        self.xlstm_cfg.model.embedding_dim, 
                                        kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1))

        if  xlstm_cfg.snn.is_use:
            self.obs_down_conv = DownConvSNN(self.xlstm_cfg.model.embedding_dim, self.enc_out_dim, config=xlstm_cfg)

        else:
            self.obs_down_conv = DownConv(self.xlstm_cfg.model.embedding_dim, self.enc_out_dim)

        
        full_act_seq_len = (xlstm_cfg.max_history*xlstm_cfg.past_img_num+xlstm_cfg.future_img_num)*xlstm_cfg.per_image_with_signal_num
        self.state_norm1 = nn.BatchNorm1d(xlstm_cfg.future_img_num * xlstm_cfg.per_image_with_signal_num)
        self.act_norm2 = nn.BatchNorm1d(xlstm_cfg.future_img_num * xlstm_cfg.per_image_with_signal_num)
        self.img_norm = nn.BatchNorm3d(xlstm_cfg.past_img_num + xlstm_cfg.future_img_num)
        self.norm_3 = nn.BatchNorm1d(self.enc_out_dim)
        if xlstm_cfg.snn.is_use:
            self.obs_last_frame_draw_feat = CSNN(T=xlstm_cfg.snn.T, channels=3, out_channel=11, config=xlstm_cfg) 
        else:
            self.obs_last_frame_draw_feat = DrawLastFrameFeature(in_channels=9, out_channels=xlstm_cfg.enc_out_dim - xlstm_cfg.per_image_with_signal_num, config=xlstm_cfg)

        self.critic = Critic(config=xlstm_cfg)
        self.shape_feature = nn.Linear(96, xlstm_cfg.z_dim * 2)
        

    def forward(self, batch: dict[str, torch.Tensor], phase:str=''):
        
        # prepare data #####################
        action_label = batch['action']
        state = batch['obs']['agent_pos']
        obs = batch['obs']['image']
        naction_label = self.act_norm2(action_label)
        nstate = self.state_norm1(state)
        nobs_full_seq = self.img_norm(obs)
        obs_seqlen = nobs_full_seq.shape[1]
        nobs_label = nobs_full_seq[:, obs_seqlen // 2:, ...]
        nobs = nobs_full_seq[:, :obs_seqlen // 2, ...]

        nobs_error = self.compute_error(nobs_full_seq)


        
        # visual seq encoder
        nobs_out = self.obs_init_conv3d(nobs_full_seq)

        if self.xlstm_cfg.visual_encoder == 'diffusion':
            
            obs_conv_out_1 = self.con3d_1(obs_out)
            obs_conv_out_2 = self.con3d_2(obs_conv_out_1)
            

            
            if not self.xlstm_cfg.olny_action_generate:
                obs_layers_out = [obs_conv_out_1, obs_conv_out_2]

                b, s, c, h, w = obs_conv_out_2.shape
                obs_conv_out_2_ = obs_conv_out_2.reshape(-1, 1, c, h, w).squeeze(1)
                self.output.obs_diff_loss = self.obs_diff(obs_conv_out_2_)
                
  
            else:
                self.output.obs_diff_loss = None

            obs_eq_hw = obs_conv_out_2_.transpose(1, 2).repeat(1, 1, 1, 2, 1)[:, :, :, :32, :]
            obs_visual_out = self.conv_lstm(obs_eq_hw) 
            b, s, c, h, w = obs_visual_out.shape
            obs_visual_out = obs_visual_out.reshape(b, 256, -1, h, w)
            
        else: # visual_encoder: xlstm
            
            obs_xlstm_out = self.obs_xlstm_layers.xlstm(nobs_out)

            obs_visual_out = soft_pool3d(self.obs_down_conv(obs_xlstm_out))

            self.output.obs_diff_loss = None
       
            
        obs_future_seq = []
        act_future_seq = []
        act_mu_list = []
        act_std_list = []
        act_log_prob_list = []
        weights_list = []
        bias_list = []
        self.output.pred_state = {}
        
        for ith in range(self.xlstm_cfg.future_img_num + 1):
            self.output.pred_state[f'{ith}'] = []
            #  action + visual frame encode
            if ith < self.xlstm_cfg.future_img_num:
                obs_last_frame = nobs_full_seq[:, ith, ...]

                obs_last_frame_feat = self.obs_last_frame_draw_feat(obs_last_frame, nobs_error[:, ith, ...])

                state_slice = nstate[:, 
                                        ith*self.xlstm_cfg.per_image_with_signal_num:(ith+1)*self.xlstm_cfg.per_image_with_signal_num,
                                        ...]
                # obs_error = torch.sigmoid((obs_next_frame - obs_last_frame).sum(1))

                vis_act_feat = torch.cat([obs_last_frame_feat, 
                                        state_slice,
                                        ], dim=1)
                if self.xlstm_cfg.act_encoder == 'causalconv':
                    act_out = self.causal_model(state_slice)
                elif self.xlstm_cfg.act_encoder == 'transformerxlstm':
                    vis_act_feat = vis_act_feat.reshape(self.xlstm_cfg.batchsize, self.xlstm_cfg.act_model_enc.embedding_dim, -1)
                    act_out = self.transbilstm(vis_act_feat)
                elif self.xlstm_cfg.act_encoder == 'xlstm':
                    vis_act_feat = vis_act_feat.reshape(self.xlstm_cfg.batchsize, self.xlstm_cfg.act_model_enc.embedding_dim, -1, 1, 1)
                    act_out = self.act_xlstm.xlstm(vis_act_feat)
                    act_out = act_out.repeat(1, 1, self.enc_out_dim // act_out.size(2) + 1, 1, 1)[:, :, :self.enc_out_dim, ...].transpose(1, 2)
            
            if self.xlstm_cfg.is_use_mam:
                fused_modal = self.modal_fusion_model([obs_visual_out, act_out])
            else:
                grip_out_temp = torch.flatten(obs_visual_out, start_dim=2)
                act_out_temp = torch.flatten(act_out, start_dim=2)
                features = torch.cat([grip_out_temp, act_out_temp], dim=-1)
                fused_modal = self.shape_feature(features)
            self.output.mu, self.output.var = gaussian_parameters(fused_modal)
            z_mix = sample_gaussian(self.output.mu, self.output.var, 
                                    device=self.device, 
                                    z_attention=self.xlstm_cfg.z_attention, SelfAttentions=self.self_attentions, training_phase=phase)
            # visualize z
    
            ### actions
            if ith < self.xlstm_cfg.future_img_num:

                z_mix_act_out = torch.cat([z_mix, act_out.squeeze(-1).squeeze(-1)], dim=-1)
                # z_mix_act_out = self.norm_3(z_mix_act_out)
                z_mix_act_out = self.act_attn(z_mix_act_out)
                act_mu, act_std, next_act, log_prob = self.act_generate(z_mix_act_out)
                act_future_seq.append(next_act)
                if act_mu is not None and act_std is not None:
                    act_mu_list.append(act_mu)
                    act_std_list.append(act_std)
                    act_log_prob_list.append(log_prob)
                self.output.pred_state[f'{ith}'].append(next_act)

            ### generate images
            # skip_layers = [xlstm_out, visual_out]
            if not self.xlstm_cfg.olny_action_generate and self.xlstm_cfg.act_encoder == 'transformerxlstm':
                next_frame = None
                obs_last_frame = nobs_full_seq[:, -self.xlstm_cfg.future_img_num-1+ith, ...]      

                if next_frame is None:
                    obs_next_frame = nobs[:, -2, ...]

                obs_error = nobs_error[:, -self.xlstm_cfg.future_img_num-1+ith, ... ]

                obs_next_frame = self.obs_img_generate(z_mix, obs_last_frame.unsqueeze(1), obs_error.unsqueeze(1), nstate)

                obs_future_seq.append(obs_next_frame)
                self.output.pred_state[f'{ith}'].append(obs_next_frame)

            # weights & bias  
            if ith < self.xlstm_cfg.future_img_num:
                weights, bias = self.critic(self.output.pred_state[f"{ith}"])
                weights_list.append(weights)
                bias_list.append(bias)
         
        if obs_future_seq != []:
            self.output.obs_future_seq = torch.cat(obs_future_seq[:self.xlstm_cfg.future_img_num], dim=1)
            self.output.obs_future_seq_more = torch.cat(obs_future_seq, dim=1)

        if act_std_list != []:
            self.output.act_mu = torch.cat(act_mu_list, dim=1)
            self.output.act_std = torch.cat(act_std_list, dim=0)
            self.output.act_log_prob = torch.cat(act_log_prob_list, dim=1)
        else:
            self.output.act_mu = None
            self.output.act_std = None
        self.output.actions = torch.cat(act_future_seq, dim=1)
        self.output.weights = torch.cat(weights_list, dim=1)
        self.output.bias = torch.cat(bias_list, dim=1)
        
        return self.output


class ComputeLIFResiduals(nn.Module):
    def __init__(self, threshold=0.5, is_use_snn_residual=False):
        super(ComputeLIFResiduals, self).__init__()
        self.threshold = threshold
        self.mem = None
        self.is_use_snn_residual = is_use_snn_residual
        
        
    def forward(self, input_):
        with torch.no_grad():
            # 计算前后相邻图像的差值
            prev_imgs = input_[:, :-1, ...]  # 前 N-1 张图像 [B, N-1, C, H, W]
            next_imgs = input_[:, 1:, ...]   # 后 N-1 张图像 [B, N-1, C, H, W]
            f_error = torch.sigmoid(next_imgs - prev_imgs)  # 残差张量 [B, N-1, C, H, W]
            b_error = torch.sigmoid(prev_imgs - next_imgs)
            if self.is_use_snn_residual:
                f_error = self.LIF(f_error)
                b_error = self.LIF(b_error)
                
            error = torch.cat([f_error, b_error], dim=2)
            
        return error

    def LIF(self, input_):      
        if self.mem is None:
            self.mem = torch.zeros(input_.shape).to(input_.device)  # 初始化膜电位

        self.mem += input_  # 累积输入
        spikes = (self.mem > self.threshold).float()  # 发放脉冲
        self.mem -= self.threshold * spikes  # 重置膜电位
        return spikes

class DrawLastFrameFeature(nn.Module):
    def __init__(self, in_channels=9, out_channels=150, config=None):
        super(DrawLastFrameFeature, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, 36, (5, 5), stride=(2, 2), padding=(2, 2), bias=True)
        self.batchnorm_1 = nn.BatchNorm2d(36)
        self.conv2d_2 = nn.Conv2d(36, 72, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.batchnorm_2 = nn.BatchNorm2d(72)
        self.conv2d_3 = nn.Conv2d(72, out_channels, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.batchnorm_3 = nn.BatchNorm2d(out_channels)
        self.flatten = nn.Flatten(start_dim=2)
        if config.data_format == 'rpy':
            self.fc = nn.Linear(49, 2, bias=True)
        elif config.data_format == 'joints':
            self.fc = nn.Linear(36, 2, bias=True)
            
        self.norm_ = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, error):
        x = torch.cat([x, error], dim=1)
        x = soft_pool2d(self.batchnorm_1(self.conv2d_1(x)))
        x = self.relu(x)
        x = soft_pool2d(self.batchnorm_2(self.conv2d_2(x)))
        x = self.relu(x)
        x = soft_pool2d(self.batchnorm_3(self.conv2d_3(x)))
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm_(x)
        return x
    
    
class ImageDiffusion:
    def __init__(self) -> Unet:
        unet_enc = Unet(
            dim=8,                # 基础维度保持不变
            channels=3,           # 修改为输入的通道数
            dim_mults=(1, 2),      # 根据输入的形状适配下采样的次数
            
        )
        # 修改 GaussianDiffusion 的参数
        self.diff_enc = GaussianDiffusion(
            unet_enc,
            image_size=(20, 32),    # 匹配输入的图像大小，需要能被2整除
            timesteps=100,         # 时间步长保持不变
            sampling_timesteps=100,  # 保持采样步长
        )
        return unet_enc

class CrossConvolution(nn.Module):
    def __init__(self, z_dim=128, in_channels=3, out_channels=1):
        super(CrossConvolution, self).__init__()
        # self.img_tac_conv = nn.Conv1d(in_channels - 1, out_channels, kernel_size=3, padding=1)
        # self.flex_conv = nn.Conv1d(in_channels - 1, out_channels, kernel_size=3, padding=1)
        self.cross_conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, multimodal_dat):
        return self.leakyrelu(self.cross_conv3d(multimodal_dat))

class SelfAttentions(nn.Module):
    def __init__(self, input_dim=256, seq_len=70, eq_output=False):
        '''
        Self attentions moudule
        param:
        input_dim: int
        seq_len: int 
        eq_output: bool, is output dim eq for input dim
        '''
        super(SelfAttentions, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.normal_modal = nn.BatchNorm1d(num_features=seq_len)
        if eq_output:
            self.fc_out = nn.Linear(input_dim, input_dim)
        else:
            self.fc_out = nn.Linear(input_dim, input_dim // 2)
         
    def forward(self, mu, var=None):
        if mu is not None and var is not None:
            x = torch.cat([mu, var], dim=-1)
        else:
            x = mu
        norm_out = self.normal_modal(x)
        x = self.fc(norm_out)
        attention_weights = F.softmax(x, dim=1) ## 
        
        attended_modalities = torch.mul(attention_weights, norm_out)
        attended_modalities = torch.add(attended_modalities, norm_out)
        out = self.fc_out(attended_modalities)
        
        return out

class MultiModalAttention(nn.Module):
    def __init__(self, z_dim=128, num_modalities=3, xlstm_cfg=None):
        super(MultiModalAttention, self).__init__()
        self.num_modalities = num_modalities
        self.z_dim = z_dim
        # magic num 
        if xlstm_cfg.both_camera_concat_over == 'c_channel' and xlstm_cfg.act_encoder == 'causalconv':
            self.fc = nn.Linear(112, 1)
        elif xlstm_cfg.both_camera_concat_over == 'w_channel' and xlstm_cfg.act_encoder == 'transformerxlstm':
            self.fc = nn.Linear(72, 1)
        elif xlstm_cfg.both_camera_concat_over == 'w_channel' and xlstm_cfg.act_encoder == 'xlstm':
            self.fc = nn.Linear(72, 1)
        else:
            raise ValueError ("keyword error")

    def forward(self, modalities):
        # Compute attention weights
        
        attention_weights = F.softmax(self.fc(modalities), dim=1) ### 
        
        attended_modalities = torch.mul(attention_weights, modalities)
        
        return attended_modalities

class MultiModalFusionModel(nn.Module):
    def __init__(self, num_modalities=3, z_dim=256, input_dim=256, is_use_cross_conv=True, xlstm_cfg=None):
        super(MultiModalFusionModel, self).__init__()
        self.num_modalities = num_modalities
        self.z_dim = z_dim * 2  # need to split mu&var, so * 2
        self.is_use_cross_conv = is_use_cross_conv
        self.seq_dim = input_dim
        # Define cross convolution layers for each modality
        self.fusion_model = CrossConvolution(in_channels=self.seq_dim, out_channels=self.seq_dim, z_dim=self.z_dim)
        # self.fusion_model.apply(init_weights)
        # Define multi-modal attention
        self.attention = MultiModalAttention(num_modalities=num_modalities, z_dim=self.seq_dim, xlstm_cfg=xlstm_cfg)
        # self.attention.apply(init_weights)
        # self.cross_conv_fusion = nn.Conv1d(self.z_dim, self.z_dim, kernel_size=3, padding=1, stride=1)
        self.normal_modal = nn.BatchNorm1d(num_features=self.seq_dim)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.flatten = nn.Flatten()
        # Fully connected layers for classification
        if xlstm_cfg.both_camera_concat_over == 'c_channel':
            self.fc1 = nn.Linear(112, self.z_dim)
        elif xlstm_cfg.both_camera_concat_over == 'w_channel' and xlstm_cfg.act_encoder == 'causalconv':
            self.fc1 = nn.Linear(16384, self.z_dim)
        elif xlstm_cfg.both_camera_concat_over == 'w_channel' and xlstm_cfg.act_encoder == 'transformerxlstm':
            self.fc1 = nn.Linear(72, self.z_dim)
        elif xlstm_cfg.both_camera_concat_over == 'w_channel' and xlstm_cfg.act_encoder == 'xlstm':
            self.fc1 = nn.Linear(72, self.z_dim)
        else:
            raise ValueError ("keyword error")
        

    def forward(self, modalities):
        '''
        Param:
        modaltities:[
            [B, S, ...],  
            ...
            ]
        '''
        # modalities: [visal, act]
        for d_cnt, d in enumerate(modalities):
            assert d.size(1) == self.seq_dim, "input shape must be [B, S, ...], and S=input_dim"
            if len(d.shape) > 3:
                d_shape = d.shape
                modalities[d_cnt] = d.reshape(d_shape[0], d_shape[1], -1)
        # optional 
        modalities[-1] = modalities[-1].repeat(1, 1, modalities[0].size(-1) // modalities[-1].size(-1) + 1)[:, :, :modalities[0].size(-1)]
        
        modality_outputs = torch.cat(modalities, dim=-1)
        batch_size, num_modal, num_features = modality_outputs.size()
        normal_result = self.normal_modal(modality_outputs)
        # Apply multi-modal attention
        attention_result = self.attention(normal_result)
        sum_result = torch.add(normal_result, attention_result)
        if self.is_use_cross_conv:
            fused_result = self.fusion_model(sum_result.unsqueeze(-1).unsqueeze(-1))
        else:
            fused_result = sum_result 
        # sum_fused_result = torch.sum(fused_result, dim=1)
        # flatten_result = self.flatten(fused_result.squeeze(-1).squeeze(-1))
        fused_result = fused_result.squeeze(-1).squeeze(-1)
        fc_out = self.fc1(fused_result)

        return fc_out


class CausalNet(nn.Module):
    def __init__(self, z_dim=256, num_img=12, initailize_weights=True, xlstm_cfg=None):
        super(CausalNet, self).__init__()
        self.z_dim = z_dim
        self.num_img = num_img
        in_channels = xlstm_cfg.past_img_num + xlstm_cfg.future_img_num
        self.causconv1 = CausalConv2D(in_channels, 32, kernel_size=(3, 3), stride=1, dilation=1)
        self.causconv2 = CausalConv2D(32, 64, kernel_size=(3, 3), stride=1, dilation=1)
        self.causconv3 = CausalConv2D(64, 64, kernel_size=(3, 3), stride=1, dilation=1)
        self.causconv4 = CausalConv2D(64, 128, kernel_size=(3, 3), stride=1, dilation=1)
        self.causconv5 = CausalConv2D(128, z_dim, kernel_size=(3, 3), stride=1, dilation=1)
            
        self.flatten = nn.Flatten()


    def forward(self, tactile):
        leaky_result1, causconv1_result1 = self.causconv1(tactile)
        leaky_result2, causconv1_result2 = self.causconv2(leaky_result1)
        leaky_result3, causconv1_result3 = self.causconv3(leaky_result2)
        leaky_result4, causconv1_result4 = self.causconv4(leaky_result3)
        leaky_result5, causconv1_result5 = self.causconv5(leaky_result4)
 
        return leaky_result5 

class TransformerXLSTM(nn.Module):
    def __init__(self, input_channels=11, embed_dim=240, hidden_dim=128, 
                 num_layers=6, num_heads=10, output_dim=16, max_history=5, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.config = config
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # 输入通道适配层（替换原Conv1d）
        if not config.is_only_transformer:
            self._xlstm = XLstmStack(self.config.act_model_enc, device=device)
        else:
            num_layers = num_layers + config.act_model_enc.num_blocks
        # if not config.is_only_transformer:
        self.input_proj = nn.Linear(input_channels, embed_dim)

        
        # Transformer Encoder (调整d_model为embed_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # BiLSTM (调整输入维度)
        # self.bilstm = nn.LSTM(
        #     input_size=embed_dim, 
        #     hidden_size=hidden_dim, 
        #     num_layers=num_layers//2,  # 双向需减半层数
        #     bidirectional=True, 
        #     batch_first=True
        # )
        
        self.norm_ = nn.BatchNorm1d(hidden_dim // 2)
        # 输出层
        self.fc = nn.Linear(config.act_model_enc.embedding_dim * 2, output_dim, bias=True)  # 双向输出合并
        self.norm_2 = nn.BatchNorm1d(config.act_model_enc.embedding_dim)

    def forward(self, x):
        """
        输入x: (batch_size, seq_len, input_channels)
        lengths: 实际有效序列长度（用于动态处理）
        """
        # 1. 输入投影
        x = self.input_proj(x)  # (B, S, embed_dim)
        if not self.config.is_only_transformer:
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self._xlstm.xlstm(x)
            x = x.squeeze(-1).squeeze(-1)
        x = self.norm_2(x)
        out = self.transformer_encoder(x)  # (B, S, embed_dim)

        # 5. 输出层  
        out = out.reshape(out.size(0), self.hidden_dim // 2, -1)
        out = self.norm_(out)
        return self.fc(out)
    
class DownConv(nn.Module):
    def __init__(self, embedding_dim, enc_out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_seq = nn.Sequential(
            nn.Conv3d(embedding_dim, 
                        embedding_dim * 2,
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1)
                                    ),
            nn.BatchNorm3d(embedding_dim * 2),
            nn.ReLU(),
            nn.Conv3d(embedding_dim * 2, 
                        enc_out_dim,
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1)),
            nn.BatchNorm3d(enc_out_dim),
            nn.ReLU())
        
    def forward(self, x):
        return self.conv_seq(x)

class ConvLstmEnc(nn.Module):
    def __init__(self, input_channels=64, output_channel=128) -> None:
        super(ConvLstmEnc, self).__init__()
        self.conv_lstm1 = ConvLSTM(input_channels=input_channels, hidden_channels=64, num_layers=1, first_flag=True)
        self.conv_lstm2 = ConvLSTM(input_channels=64, hidden_channels=128, num_layers=1, stride=2, size_flag='up')
        self.conv_lstm3 = ConvLSTM(input_channels=128, hidden_channels=output_channel, num_layers=1)

        
    def forward(self, x):
        h, c = self.conv_lstm1(x)
        h, c = self.conv_lstm2(h, c)
        h, c = self.conv_lstm3(h, c)
        return self.frame_to_stream(h)
    
    def frame_to_stream(self, in_list):
        temp_state = [temp.unsqueeze(2) for temp in in_list]
        cat_result = torch.cat(temp_state, dim=2)
        return cat_result
    
def get_images_error(last_frame:torch.Tensor, curr_frame:torch.Tensor, seq:int):
    error_pred_label = F.sigmoid(last_frame - curr_frame)  
    error_label_pred = F.sigmoid(curr_frame - last_frame)
    error_all = torch.cat([error_label_pred, error_pred_label], dim=2)
    # img_show(last_frame[-1][-1], error_pred_label[-1][-1], error_label_pred[-1][-1])
    return error_all

from matplotlib import pyplot as plt
import numpy as np
def img_show(*images):
    """
    显示多个RGB张量的图片

    参数:
    *images: 任意数量的RGB张量
    格式：
    必须是[channel, h, w]或者[channel, num_img, h, w]
    不带batchsize
    """
    num_images = len(images)
    if num_images <= 1:
        images = images[0]
        num_images = images.shape[1]

    # 设置子图的行和列
    rows = 1
    cols = num_images

    # 创建一个新的图形
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # 如果只有一张图片，将axes转换为一个包含单个元素的列表
    if num_images == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # 将RGB张量的值限制在0到1之间
        if num_images <= 1:
            image_data = np.clip(images[i].permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
        else:
            if images[i].size(0) == 3:
               image_data = np.clip(images[i][:, :, :].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) 
            elif images[i].size(0) >= 3 and images[i].size(0) % 3 == 0:
                img = torch.cat(torch.split(images[i], 3), dim=1)
                image_data = np.clip(img.permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
            
            
        # 显示图片
        ax.imshow(image_data)
        ax.axis('off')

    plt.show()