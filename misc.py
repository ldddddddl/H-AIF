import torch
# from box import Box 
from collections import OrderedDict
# from publisher.jetmax_kinematics import forward_kinematics, inverse_kinematics
import random
import os
import numpy as np
from torch.autograd import Variable
from math import exp
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from collections import deque

class VariableContainer:
    def __init__(self):
        # 初始化一个统一的存储字典
        object.__setattr__(self, "_attributes", OrderedDict())

    def __setattr__(self, name, value):
        if name == "_attributes":
            # 允许设置内部字典属性
            object.__setattr__(self, name, value)
        else:
            # 将所有其他属性存入字典
            self._attributes[name] = value

    def __getattr__(self, name):
        # 从字典中获取属性
        attributes = object.__getattribute__(self, "_attributes")
        if name in attributes:
            return attributes[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __delattr__(self, name):
        if name in self._attributes:
            del self._attributes[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SegmentBuffer:
    """维护固定长度的历史片段队列，自动填充或截断"""
    def __init__(self, segment_dim:list=[8, 50, 11], config=None):
        self.config = config
        self.max_history = config.max_history
        self.segment_dim = segment_dim
        self.buffer = deque(maxlen=self.max_history)
        self.init_buffer(self.max_history)
        self.cnt = 0
    def init_buffer(self, maxlen):
        for _ in range(maxlen):
            temp = torch.tensor(
                [0.0, 0.008055363, 0.130413211, 0.408738913, 0.13037428, 0.131727096,
                -0.271861495, 0.409781488, -0.278362593, -0.278366361, 1],
                dtype=torch.float32, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
                ).reshape(1, 1, -1).repeat(self.config.batchsize, self.config.past_img_num*self.config.per_image_with_signal_num, 1)
            self.buffer.append(temp)
        
    def add_segment(self, segment, add_side='r'):
        """
        添加新片段并返回处理后的序列
        """
        #要加大max_history， 最好在这实现
        if add_side == 'r':
            self.buffer.append(segment)
        else:
            self.buffer.appendleft(segment)
            
        self.cnt += 1
        return {
            "length":[self.cnt, -1],
            "target":torch.cat(list(self.buffer), dim=1)}  # (max_history, segment_dim)

def update_ssim_psnr(grip_results_total, side_results_total, valid_avg:AverageMeter):
    grip_ssim_frames_res_np, grip_psnr_frames_res_np, grip_results = grip_results_total
    side_ssim_frames_res_np, side_psnr_frames_res_np, side_results = side_results_total
    valid_avg.grip_ssim_mean.update(grip_results['ssim_mean'])
    valid_avg.grip_ssim_var.update(grip_results["ssim_var"])
    valid_avg.grip_psnr_mean.update(grip_results["psnr_mean"])
    valid_avg.grip_psnr_var.update(grip_results["psnr_var"])
    
    valid_avg.side_ssim_mean.update(side_results["ssim_mean"])
    valid_avg.side_ssim_var.update(side_results["ssim_var"])
    valid_avg.side_psnr_mean.update(side_results["psnr_mean"])
    valid_avg.side_psnr_var.update(side_results["psnr_var"])
    
    valid_avg.grip_ssim_frames.update(grip_ssim_frames_res_np)
    valid_avg.grip_psnr_frames.update(grip_psnr_frames_res_np)
    
    valid_avg.side_ssim_frames.update(side_ssim_frames_res_np)
    valid_avg.side_psnr_frames.update(side_psnr_frames_res_np)
    
    

class MatrixSSIMPSNR:
    def __init__(self, data_range=1.0):
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.ssim_compute = SSIM(data_range=data_range).to(device)
        self.psnr_compute = PSNR(data_range=data_range).to(device)
    
    def compute(self, predict:torch.Tensor, label:torch.Tensor):
        '''
        Input:
        predict shape(shape > 4): [batch, seq_len, c, h, w] 
        predict shape(shape = 4): [batch, c, h, w] 
        
        Return:
        ssim_frames_result: np.array
        psnr_frames_result: np.array
        results_mean: dict
        '''
        results_mean = {}
        if len(predict.shape) > 4:
            ssim_frames_result = [self.ssim_compute(predict[:, s, ...], label[:, s, ...]).item() for s in range(predict.size(1))]
            psnr_frames_result = [self.psnr_compute(predict[:, s, ...], label[:, s, ...]).item() for s in range(predict.size(1))]
            ssim_frames_res_np = np.array(ssim_frames_result, dtype=np.float16)
            results_mean["ssim_mean"] = np.mean(ssim_frames_result)
            results_mean["ssim_var"] = np.var(ssim_frames_result)
            
            psnr_frames_res_np = np.array(psnr_frames_result, dtype=np.float16)
            results_mean["psnr_mean"] = np.mean(psnr_frames_result)
            results_mean["psnr_var"] = np.var(psnr_frames_result)
            
        elif len(predict.shape) == 4:
            ssim_frames_result = self.ssim_compute(predict, label)
            psnr_frames_result = self.psnr_compute(predict, label)
            
        return [ssim_frames_res_np, psnr_frames_res_np, results_mean]
        

class TorchMetricsWrapper:
    def __init__(self, device='cuda'):
        self.precision_metric = Precision(task='multiclass', average='macro', num_classes=100).to(device)
        self.recall_metric = Recall(task='multiclass', average='macro', num_classes=100).to(device)
        self.f1_metric = F1Score(task='multiclass', average='macro', num_classes=100).to(device)
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """更新指标"""
        self.precision_metric.update(preds, labels)
        self.recall_metric.update(preds, labels)
        self.f1_metric.update(preds, labels)
    
    def compute(self) -> dict:
        """计算所有指标"""
        return {
            'precision': self.precision_metric.compute().item(),
            'recall': self.recall_metric.compute().item(),
            'f1': self.f1_metric.compute().item()
        }
    
    def reset(self):
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()



class InitAverager:
    def __init__(self):
        self.actions_loss = AverageMeter()
        self.new_actions_loss = AverageMeter()
        self.grip_frames_loss = AverageMeter()
        self.side_frames_loss = AverageMeter()
        self.image_kl_loss = AverageMeter()
        self.act_kl_loss = AverageMeter()
        self.sucker_loss = AverageMeter()
        self.gdl_loss = AverageMeter()
        self.sucker_pred_acc = AverageMeter()
        self.actions = AverageMeter() 
        self.grip_diff_loss = AverageMeter() 
        self.side_diff_loss = AverageMeter() 
        self.infer_time = AverageMeter()
        
        self.grip_ssim_frames = AverageMeter()
        self.grip_ssim_mean = AverageMeter()
        self.grip_ssim_var = AverageMeter()
        
        self.side_ssim_mean = AverageMeter()
        self.side_ssim_var = AverageMeter()
        self.side_ssim_frames = AverageMeter()
        
        self.grip_psnr_mean = AverageMeter()
        self.grip_psnr_var = AverageMeter()
        self.grip_psnr_frames = AverageMeter()
        
        self.side_psnr_mean = AverageMeter()
        self.side_psnr_var = AverageMeter()
        self.side_psnr_frames = AverageMeter()
        
        self.act_jerk_mean = AverageMeter()
        self.act_jerk_nrom = AverageMeter()
        self.act_azc = AverageMeter()
        self.act_pi = AverageMeter()
        self.act_hf = AverageMeter()
        self.act_ea = AverageMeter()

        self.new_act_jerk_mean = AverageMeter()
        self.new_act_jerk_nrom = AverageMeter()
        self.new_act_azc = AverageMeter()
        self.new_act_pi = AverageMeter()
        self.new_act_hf = AverageMeter()
        self.new_act_ea = AverageMeter()
        
        self.label_act_jerk_mean = AverageMeter()
        self.label_act_jerk_nrom = AverageMeter()
        self.label_act_azc = AverageMeter()
        self.label_act_pi = AverageMeter()
        self.label_act_hf = AverageMeter()
        self.label_act_ea = AverageMeter()
        
        # self.valid_act_loss = AverageMeter()
        # self.valid_frames_loss = AverageMeter()
        # self.valid_kl_loss = AverageMeter()
        # self.valid_sucker_loss = AverageMeter()
        # self.valid_gdl_loss = AverageMeter()
        # self.valid_sucker_pred_acc = AverageMeter()

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def process_image(img:torch.Tensor, H, W):
    '''
    返回一个指定大小的RGB图像
    '''
    return Image.fromarray(np.uint8(norm_to_rgb(img.cpu()))).resize((W, H))

def save_images(best_results, batch: int=0, shape: str = 'w_channel', save_path: str = r'/results', epc: int = 0):
    if shape == 'w_channel':
        
        B, S, C, H, W  = best_results.side_frame_labels.shape
        
        pred_gripper_frames = best_results.grip_upsample_frame_pred[batch].transpose(1, -1)
        pred_side_frames = best_results.side_upsample_frame_pred[batch].transpose(1, -1)
        labels_gripper_frames = best_results.grip_frame_labels[batch].transpose(1, -1)
        labels_side_frames = best_results.side_frame_labels[batch].transpose(1, -1)
        
        with ThreadPoolExecutor() as executor:
            pred_gripper_results = list(executor.map(process_image, pred_gripper_frames, [H] * (S + 1), [W] * (S + 1)))
            pred_side_results = list(executor.map(process_image, pred_side_frames, [H] * (S + 1), [W] * (S + 1)))
            
            label_gripper_results = list(executor.map(process_image, labels_gripper_frames, [H] * S, [W] * S))
            label_side_results = list(executor.map(process_image, labels_side_frames, [H] * S, [W] * S))
            # 单独保存每张图像
        for i in range(S + 1):
            # 保存预测图像
            pred_gripper_results[i].save(f"{save_path}/epc{epc+1}_pred_gripper_batch{batch}_{i}.png")
            pred_side_results[i].save(f"{save_path}/epc{epc+1}_pred_side_batch{batch}_{i}.png")
            # 保存标签图像
            if i < S:
                label_gripper_results[i].save(f"{save_path}/epc{epc+1}_label_gripper_batch{batch}_{i}.png")
                label_side_results[i].save(f"{save_path}/epc{epc+1}_label_side_batch{batch}_{i}.png")
    # 计算画布大小，两行的总高度
    total_width = 2 * S * W
    total_height = 2 * H

    # 创建一个新的画布
    new_image = Image.new('RGB', (total_width, total_height))

    if shape == 'c_channel' or shape == 'w_channel':
        x_offset = 0
        for img in zip(pred_gripper_results, pred_side_results):
            new_image.paste(img[0], (x_offset, 0))
            new_image.paste(img[1], (x_offset + W * S, 0))
            x_offset += W

        x_offset = 0
        for img_ in zip(label_gripper_results, label_side_results):
            new_image.paste(img_[0], (x_offset, H))
            new_image.paste(img_[1], (x_offset + W * S, H))
            x_offset += W
 
    new_image.save(f"{save_path}/{epc + 1}_batch{batch}.png")




def split_list(input_list:list, ratio:float):
    '''
    Return:
    tuple[[ratio], [1-ratio]]
    '''
    
    # 计算要抽取的元素数量
    split_size = int(len(input_list) * ratio)
    
    # 随机抽取指定比例的元素
    selected_items = random.sample(input_list, split_size)
    
    # 剩余的元素
    remaining_items = [item for item in input_list if item not in selected_items]
    
    return selected_items, remaining_items

def set_seed(manual_seed):
    if manual_seed == -1:
        manual_seed = random.randint(0, 2**32 - 1)
    # 设置Python内置random库的随机种子
    random.seed(manual_seed)

    # 设置Numpy的随机种子
    np.random.seed(manual_seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(manual_seed)

    # 如果使用GPU，还需要设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)  # 如果有多个GPU

    # 确保使用确定性的卷积操作（在某些情况下可能会影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    return manual_seed

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def norm_to_rgb(in_tensor, shape='CHW'):
    '''
    input shape: [c, h, w]
    '''
    assert len(in_tensor.shape) == 3, "the shape must be eq 3"
    if shape == "CHW":
        try:
            img_data_nextframe = in_tensor.cpu().detach()
            img_data_trans = img_data_nextframe.numpy()
        except:
            img_data_trans = in_tensor, (1, 2, 0)
    else:
        try:
            img_data_trans = img_data_nextframe.numpy()
        except:
            img_data_trans = in_tensor.numpy()

    img_data_nextframe_scaled = (img_data_trans - img_data_trans.min()) / (img_data_trans.max() - img_data_trans.min() + 1e-10) * 255.0
    img_data_nextframe_uint8 = img_data_nextframe_scaled.astype('uint8')
    return img_data_nextframe_uint8


def save_act(writer, act_pred, act_label, is_prediction=False):
    # 构建 DataFrame
    import pandas as pd
    rows = []
    for (episode, act_pred_seq), (_, act_label_seq) in zip(act_pred.items(), act_label.items()):
        for (seq_p, acts_pred), (seq_l, acts_label) in zip(act_pred_seq.items(), act_label_seq.items()):
            row_p = [episode, seq_p]  # 每行以 Episode 和 Sequence 开头
            row_l = [episode, seq_l]  # 每行以 Episode 和 Sequence 开头
            max_len = max(len(v) for v in acts_pred.values())  # 找到最长动作序列
            for i in range(len(acts_pred)):
                row_p.extend([acts_pred.get(f"act_{i}", [None] * max_len)[c] for c in range(max_len)])
                row_l.extend([acts_label.get(f"act_{i}", [None] * max_len)[c] for c in range(max_len)])
            rows.append(row_p)
            rows.append(row_l)

    # 创建表头
    max_steps = max(len(v) for seq in act_pred.values() for actions in seq.values() for v in actions.values())

    header = ["Episode", "Sequence"] + [f"act{i + 1}_{a + 1}" for i in range(len(acts_pred)) for a in range(max_steps)]

    # 转换为 DataFrame
    result_df = pd.DataFrame(rows, columns=header)

    result_df.to_excel(writer, sheet_name='actions', index=False)

        
def tens2act(in_tensor, epc=0, name=''):
    '''
    split 9 joint from batch size
    '''

    # split_act = torch.split(in_tensor, 1, dim=0)
    joint_act = torch.cat(in_tensor, dim=0).squeeze(0)
    act_dict = {
        f'{name}_epc{epc}_act_{joint + 1}':joint_act[:, joint].tolist() for joint in range(joint_act.size(-1))
    }   
    return act_dict


def repeat_data(datas, batchsize):
    label_list = []
    for data in datas:
        data_len = len(data.shape)
        pad = (1, ) * (data_len - 1)
        label_list.append(data.repeat(batchsize, *pad))
    return label_list