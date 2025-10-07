import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from misc import VariableContainer, norm_to_rgb, AverageMeter
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import numpy as np
import scipy.stats

class ComputeLosses(nn.Module):
    
    '''
    forward method:
    input: 
    pred_act: Tensor
    pred_sucker: Tensor
    pred_frames: Tensor
    labels: list[Tensor, ...]  --> [sucker_labels, joints_pos_labels, gripper_frame_labels, side_frame_labels]
    
    return:
    losses: 
    '''
    def __init__(self, device, config, act_loss='mse'):
        super(ComputeLosses, self).__init__()
        self.config = config
        self.device = device
        self.act_loss = act_loss
        self.gdl_loss = GradientDifferenceLoss()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss(   )
        
        
        
    def forward(self, pred_results:VariableContainer, labels:list, writer:SummaryWriter, epoch:int, avg_:AverageMeter, phase:str=None):
        loss_results = VariableContainer()
        sucker_labels, joints_pos_labels, gripper_frame_labels, side_frame_labels, _ = [label.to(self.device) for label in labels]
        if self.config.is_diff_generate_act:
            actions_loss = pred_results.actions_loss
        elif self.act_loss == 'mse':
            actions_loss = self.mse_loss(pred_results.actions, joints_pos_labels)
        elif self.act_loss == 'l4':
            actions_loss = self.l4_norm(pred_results.actions, joints_pos_labels)
            
        # act_kl = self.kl_dist(pred_results.actions, joints_pos_labels)
        # act_kl = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        if not self.config.olny_action_generate:
            if self.config.both_camera_concat_over == 'c_channel':
                future_frame_labels = torch.cat([gripper_frame_labels, side_frame_labels], dim=-1).transpose(2, -1)
            elif self.config.both_camera_concat_over == 'w_channel':
                # future_frame_labels = torch.cat([gripper_frame_labels, side_frame_labels], dim=2).transpose(2, -1)
                pass
            else:
                raise ValueError('keyword error')
            grip_frames_loss, loss_results.grip_upsample_frame_pred = self.realEPE(pred_results.grip_future_seq, gripper_frame_labels) 
            side_frames_loss, loss_results.side_upsample_frame_pred = self.realEPE(pred_results.side_future_seq, side_frame_labels) 
        else:
            grip_frames_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            side_frames_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            loss_results.grip_upsample_frame_pred = None
            loss_results.side_upsample_frame_pred = None            
        # gdl_loss = self.gdl_loss(pred_results.future_seq, future_labels)
        if phase == 'inference':
            phase_alpha_kl = 0.0
        elif phase == 'add_kl':
            phase_alpha_kl = (epoch + 1 - self.config.epochs * 0.5) / (self.config.epochs * 0.25)
        else:
            phase_alpha_kl = 1.0
            
        image_kl = torch.mean(self.kl_normal(pred_results.mu, pred_results.var))
        if pred_results.act_mu is not None or pred_results.act_std is not None:
            act_kl = torch.mean(self.kl_normal(pred_results.act_mu, pred_results.act_std))
        else:
            act_kl = torch.tensor([0], dtype=torch.float32, device=self.device)
        # sucker_loss = sum([self.cross_entropy(pred_results.sucker[:, s, :], sucker_labels[:, s]) for s in range(self.config.frames_seq)]) / self.config.frames_seq

        sucker_loss = self.cross_entropy(pred_results.sucker.transpose(1, -1), sucker_labels)
        
        # activate inference
        state_loss_ratio = F.sigmoid(sucker_loss + actions_loss + grip_frames_loss + side_frames_loss)
        new_action = (state_loss_ratio * pred_results.weights + 1.)     \
                    * pred_results.actions                              \
                    + pred_results.bias
        new_action = new_action.clamp(-3.0, 3.0)
        loss_results.new_actions = new_action
        critic_loss = self.mse_loss(new_action, joints_pos_labels)
        # total loss
        loss_results.losses = (actions_loss * self.config.alpha_loss.actions
                    + critic_loss * self.config.alpha_loss.actions
                    + grip_frames_loss * self.config.alpha_loss.frames
                    + side_frames_loss * self.config.alpha_loss.frames
                    + image_kl * self.config.alpha_loss.kl * phase_alpha_kl
                    + act_kl * self.config.alpha_loss.kl
                    + sucker_loss * self.config.alpha_loss.sucker
                    + (pred_results.grip_diff_loss if pred_results.grip_diff_loss is not None else torch.tensor([0], dtype=torch.float32, device=self.device))
                    + (pred_results.side_diff_loss if pred_results.side_diff_loss is not None else torch.tensor([0], dtype=torch.float32, device=self.device))
                    # + act_kl * self.config.alpha_loss.kl
                    # + gdl_loss * self.config.alpha_loss.gdl
            ).requires_grad_(True)
         
        avg_.actions_loss.update(actions_loss.item())
        avg_.new_actions_loss.update(critic_loss.item())
        avg_.grip_frames_loss.update(grip_frames_loss.item())
        avg_.side_frames_loss.update(side_frames_loss.item())
        avg_.image_kl_loss.update(image_kl.item())
        avg_.act_kl_loss.update(act_kl.item())
        avg_.sucker_loss.update(sucker_loss.item())
        avg_.grip_diff_loss.update(pred_results.grip_diff_loss.item() if pred_results.grip_diff_loss is not None else 0)
        avg_.side_diff_loss.update(pred_results.side_diff_loss.item() if pred_results.side_diff_loss is not None else 0)
        # avg_.gdl_loss.update(gdl_loss.item())
        loss_results.gripper_frame_labels = gripper_frame_labels
        loss_results.side_frame_labels = side_frame_labels
        loss_results.actions_loss = actions_loss.item()
        loss_results.critic_loss = critic_loss.item()
        return loss_results



    def EPE(self, input_flow, target_flow, device, sparse=False, mean=True):
        # torch.cuda.init()

        EPE_map = torch.norm(target_flow.cpu() - input_flow.cpu(), 2, (1, 2, 3, 4)) # 2: [8, 8, 112, 112]
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            #target_flow[:, 0] == 0判断x方向上为0的像素点target_flow[:, 1]同理 
            #两个都为0时，标记为1
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
            EPE_map = EPE_map[~mask.data]
        if mean:
            epe_map_result = EPE_map.mean().cuda()
            return epe_map_result
        else:
            return (EPE_map.sum() / batch_size).cuda()


    def realEPE(self, output, target, device='cuda', sparse=False):
        b, d, n, h, w = target.size()

        # upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
        if output.shape == target.shape:
            upsampled_output = output
        else:
            upsampled_output = nn.functional.interpolate(output, size=(n, h, w), mode="trilinear", align_corners=False)
        return self.EPE(upsampled_output, target, device, sparse, mean=True), upsampled_output


    def kl_normal(self, qm, qv):
        # normal gausian 
        # pm = torch.nn.Parameter(
        #     torch.zeros(self.xlstm_cfg.batchsize, self.xlstm_cfg.z_dim))
        # pv = torch.nn.Parameter(
        #     torch.ones(self.xlstm_cfg.batchsize, self.xlstm_cfg.z_dim))
        
        pm = torch.zeros_like(qm)
        pv = torch.ones_like(qv)
        
        element_wise = 0.5 * (
            torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
        )
        kl = torch.sum(element_wise, dim=-1)
        return kl



    def correlation_loss(self, img_encoded, tactile_encoded, flex_encoded=None, is_mean=True):
        mean_img = torch.mean(img_encoded, dim=-1)
        mean_tactile = torch.mean(tactile_encoded, dim=-1)
        if flex_encoded != None:
            mean_flex = torch.mean(flex_encoded, dim=-1)
        corr_img_tactile = torch.div(
            torch.sum(
                torch.mul(torch.sub(img_encoded, mean_img.unsqueeze(-1)), torch.sub(tactile_encoded, mean_tactile.unsqueeze(-1))), dim=-1
            ),
            torch.sqrt(
                torch.mul(torch.sum(torch.pow(torch.sub(img_encoded, mean_img.unsqueeze(-1)), 2) , -1), torch.sum(torch.pow(torch.sub(tactile_encoded, mean_tactile.unsqueeze(-1)), 2) , -1))
            )
        )
        if is_mean:
            corr_result =  torch.mean(corr_img_tactile)
        else:
            corr_result = torch.sum(corr_img_tactile, dim=-1)
        
        return corr_result
    
    def l4_norm(self, input:torch.Tensor, target:torch.Tensor):
        return torch.norm(input.cpu() - target.cpu(), 2, (1, 2)).sum().mean()
    
    def kl_dist(self, predict:torch.Tensor, target:torch.Tensor, epsilon=1e-5):
        predict_nonneg = (predict + 100.0) / 200.0 # [-1,1] -> [0,1]
        target_nonneg = (target + 100.0) / 200.0
        qx = (predict_nonneg) / (torch.sum(predict_nonneg) + epsilon * predict.size(-1))
        py = (target_nonneg) / (torch.sum(target_nonneg) + epsilon * target.size(-1))
    
        kl_ = torch.sum(py * torch.log(py / qx + epsilon))
        return kl_

    def state_mse(self, predict_results, labels):
        sucker, action, gripper, side, _ = labels
        seq_len = sucker.shape(-1)
        for i in range(0, seq_len, self.config.per_image_with_signal_num):
            if self.config.olny_action_generate:
                predict_results.pred_state
    
    
class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, input_image, target_image):
        if input_image.shape != target_image.shape:
            b, s, c, h, w = target_image.shape
            input_image = nn.functional.interpolate(input_image, size=(c, h, w), mode="trilinear", align_corners=False)
        
        # Calculate gradients for both input and target images
        input_gradients_x = torch.abs(input_image[:, :, :, :, :-1] - input_image[:, :, :, :, 1:])
        input_gradients_y = torch.abs(input_image[:, :, :, :-1, :] - input_image[:, :, :, 1:, :])

        target_gradients_x = torch.abs(target_image[:, :, :, :, :-1] - target_image[:, :, :, :, 1:])
        target_gradients_y = torch.abs(target_image[:, :, :, :-1, :] - target_image[:, :, :, 1:, :])

        # Calculate the gradient difference loss
        gdl_loss = torch.sum(torch.sum(torch.abs(input_gradients_x - target_gradients_x), dim=(1, 2, 3, 4)) +
                             torch.sum(torch.abs(input_gradients_y - target_gradients_y), dim=(1, 2, 3, 4)))

        return gdl_loss