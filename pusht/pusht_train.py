import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from joint_states_process import JointStateRead
from model.pustht_models import ActNet, img_show
# from misc import random_target, TensorContainer
# sys.path.append(r"/home/ubuntu/Desktop/action_generation/publisher")
# from publisher import publisher
# from publisher.go_home import go_home
# from publisher.act_publisher import joint_publisher, publisher_callback
# import rospy
import time
from jmdataload import  CreateDatasets, pad_to_batch_size
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from pusht_losses import ComputeLosses, GradientDifferenceLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pusht_misc import InitAverager, set_seed, split_list, save_images, save_checkpoint, VariableContainer, save_act, SegmentBuffer, TorchMetricsWrapper, MatrixSSIMPSNR, update_ssim_psnr, tens2act
import pandas as pd
import time, os
from datetime import datetime
from multiTaskoptimizer import MultiTaskOptimizer
from spikingjelly.activation_based import functional
from pusht_image_dataset import PushTImageDataset
import hydra
from pusht_base_dataset import BaseImageDataset
from script.normalizer import LinearNormalizer

ERROR = ''
def main(xlstm_cfg: DictConfig):
    global ERROR
    seed = set_seed(xlstm_cfg.random_seed)
    xlstm_cfg.random_seed = seed
    device ="cuda" if torch.cuda.is_available() else "cpu"
    is_use_cuda = torch.cuda.is_available()
    model = ActNet(xlstm_cfg, is_use_cuda=is_use_cuda, device=device).to(device=device)

    model.apply(init_weights)
    compute_losses = ComputeLosses(device=device, config=xlstm_cfg).to(device=device)
    linear_norm = LinearNormalizer().to(device)
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=xlstm_cfg.lr,
        betas=[0.9, 0.99],
        weight_decay = 1E-4
        )
    if xlstm_cfg.is_resume:
        checkpoint = torch.load(xlstm_cfg.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model.apply(init_weights)
        
    writer = ''
    
    ctime = time.ctime()      
    ctime_datetime = datetime.strptime(ctime, "%a %b %d %H:%M:%S %Y")
    formatted_datetime = ctime_datetime.strftime("%y-%m-%d-%H-%M-%S")
    
    pwd = os.getcwd()

    if not os.path.exists(f"{pwd}/{xlstm_cfg.results_path}/{formatted_datetime}"):
        os.mkdir(f"{xlstm_cfg.results_path}/{formatted_datetime}")
        os.mkdir(f"{xlstm_cfg.results_path}/{formatted_datetime}/check_point")
        
    results_dict = {}
    actions_prediction, new_actions_prediction, total_actions = {}, {}, {}
    metrix_results, metrix_frames_result = {}, {}
    
    # configure dataset
    dataset: BaseImageDataset
    dataset = PushTImageDataset(**xlstm_cfg.task.dataset)
    # assert isinstance(dataset, BaseImageDataset)
    normalizer = dataset.get_normalizer()
    linear_norm.load_state_dict(normalizer.state_dict())
    train_dataloader = DataLoader(dataset, **xlstm_cfg.dataloader)

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_indices = val_dataset.get_val_indices()
    val_indices_list = [i[-1] for i in val_indices]
    val_indices_list = list(set(val_indices_list))
    val_dataloader = DataLoader(val_dataset, **xlstm_cfg.val_dataloader)

    try:
    # if True:
        for epc in range(xlstm_cfg.epochs):
            
            train_avg = InitAverager()
            valid_avg = InitAverager()
            
            train_phase = ''
            use_tired_training = xlstm_cfg.is_use_three_phase_train
            if use_tired_training:
                if epc / xlstm_cfg.epochs < 0.25 and use_tired_training:
                    train_phase = 'generate'
                elif 0.25 <= epc / xlstm_cfg.epochs < 0.5 and use_tired_training:
                    train_phase = 'inference'
                elif 0.5 <= epc / xlstm_cfg.epochs < 0.75 and use_tired_training:
                    train_phase = 'add_kl' 
                elif 0.75 <= epc / xlstm_cfg.epochs and use_tired_training:
                    train_phase = 'full_training'
            else:
                train_phase = None
            # train

            _ = train(model, optimizer, train_dataloader, device, compute_losses, writer, linear_norm, epc, epc, len(train_dataloader), train_avg, phase=train_phase) 

            # valid
            best_results = valid(model, optimizer, val_dataloader, device, compute_losses, writer, linear_norm, epc, epc, len(val_dataloader), valid_avg, val_indices_list, phase=train_phase)

            results_dict[f"epoch_{epc + 1}"] = {
                'train_actions_loss':train_avg.actions_loss.avg,
                'train_new_actions_loss':train_avg.new_actions_loss.avg,
                'train_grip_frames_loss':train_avg.grip_frames_loss.avg,
                'train_image_kl_loss':train_avg.image_kl_loss.avg,
                'train_act_kl_loss': train_avg.act_kl_loss,
                ## 
                'valid_actions_loss':valid_avg.actions_loss.avg,
                'valid_new_actions_loss':valid_avg.new_actions_loss.avg,
                'valid_grip_frames_loss':valid_avg.grip_frames_loss.avg,
                'valid_image_kl_loss':valid_avg.image_kl_loss.avg,
                'valid_act_kl_loss': valid_avg.act_kl_loss,
                'inference_time': valid_avg.infer_time.avg,
                }
            
            if best_results.total_actions is not None:
                actions_prediction = {**actions_prediction, **best_results.old_actions}  # old/previous action
                total_actions = {**total_actions, **best_results.total_actions} # new action
            if not xlstm_cfg.olny_action_generate:
                metrix_results[f"epoch_{epc + 1}"] = {
                    "grip_ssim_meam":valid_avg.grip_ssim_mean.avg,
                    "grip_ssim_var":valid_avg.grip_ssim_var.avg,
                    "grip_psnr_meam":valid_avg.grip_psnr_mean.avg,
                    "grip_psnr_var":valid_avg.grip_psnr_var.avg,

                }
                #  ssim 
                metrix_frames_result[f"epoch_{epc + 1}"] = {}
                
                metrix_grip_ssim_frame = {
                    f'grip_ssim_frame_{f_cnt+1}':f for f_cnt, f in enumerate(valid_avg.grip_ssim_frames.avg)
                    }
                metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_grip_ssim_frame)

                metrix_grip_psnr_frame = {
                    f'grip_psnr_frame_{f_cnt+1}':f for f_cnt, f in enumerate(valid_avg.grip_psnr_frames.avg)
                    }
                metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_grip_psnr_frame)

                if epc / xlstm_cfg.epochs >= (1.0 - xlstm_cfg.images_save_rate):

                    for b in range(xlstm_cfg.batchsize*xlstm_cfg.per_episode_to_batch_num):
                        save_images(best_results, batch=b, save_path=os.path.join(pwd, xlstm_cfg.results_path, formatted_datetime), epc=epc, shape=xlstm_cfg.both_camera_concat_over)

    except Exception as error:
        ERROR = error
        print(ERROR)
    finally:
        
        writer = pd.ExcelWriter(os.path.join(pwd, xlstm_cfg.results_path,formatted_datetime, formatted_datetime + '.xlsx'), engine='xlsxwriter')    
        results_pd = pd.DataFrame.from_dict(results_dict, orient='index')
        results_act = pd.DataFrame.from_dict(actions_prediction, orient='index')
        # results_new_act = pd.DataFrame.from_dict(new_actions_prediction, orient='index')
        total_actions_pd = pd.DataFrame.from_dict(total_actions, orient='index')
        metrix_frames_result_pd = pd.DataFrame.from_dict(metrix_frames_result, orient='index')
        metrix_results_pd = pd.DataFrame.from_dict(metrix_results, orient='index')
        results_pd.reset_index(inplace=True)
        results_pd.rename(columns={'index': 'Epochs'}, inplace=True)
        results_pd.to_excel(writer, sheet_name='results', index=False)
        results_act.to_excel(writer, sheet_name='old_actions', index=True)
        # results_new_act.to_excel(writer, sheet_name='new_actions', index=True)
        total_actions_pd.to_excel(writer, sheet_name='total_actions', index=True)
        metrix_frames_result_pd.to_excel(writer, sheet_name='frames_metrix_results', index=True)
        metrix_results_pd.to_excel(writer, sheet_name='metrix_results', index=True)
        # save_act(writer, actions_predict, actions_label, is_prediction=True)
        writer.close()
        OmegaConf.save(xlstm_cfg, os.path.join(pwd, xlstm_cfg.results_path, formatted_datetime, 'config.yaml'))
        save_checkpoint({
                    'epoch': epc + 1,
                    'state_dict': model.state_dict(),
                    'model':model,
                    'optimizer' : optimizer.state_dict(),
                    'error':ERROR
                }, checkpoint='', filename=f'results/{formatted_datetime}/check_point/model_{formatted_datetime}.pth.tar')

def train(model, optimizer, dataloader, device, compute_losses, writer, linear_norm, epc, data_cnt, total_data, train_avg, phase=None):
    # seg_buffer = SegmentBuffer(config=xlstm_cfg)
    model: ActNet
    model.train()
    for batch_idx, (batch, index) in enumerate(dataloader):

        if batch['obs']['image'].size(0)  < xlstm_cfg.batchsize:
            continue
        batch['obs'] = linear_norm.normalize(batch['obs'])
        batch['action'] = linear_norm['action'].normalize(batch['action'])
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        _time = time.time()
        output = model(batch, phase)
        loss_results = compute_losses(output, batch, writer, epc, train_avg, phase)
        # compute acc

        optimizer.zero_grad(set_to_none=True)   
        loss_results.losses.backward()
        optimizer.step()
        _run_time = time.time() - _time 
        
        print(f"\tPhase: training | Epc: {epc}/{xlstm_cfg.epochs} | Total: {batch_idx}/{len(dataloader)} | losses: {loss_results.losses.item():.2f} | training time: {_run_time:.4f}", end='\r')
        # if batch_idx > 1:
        #     break

    return loss_results

def valid(model, optimizer, dataloader, device, compute_losses, writer, linear_norm, epc, data_cnt, total_data, valid_avg, val_indices_list=None, phase=None):
    model.eval()
    best_loss = float('inf')
    timestamp = 0
    is_best_flag = None
    best_results = VariableContainer()
    compute_ssimpsnr = MatrixSSIMPSNR()
    old_pred_action = []
    pred_action, label_action = [], []
    pred_grip_images, label_grip_images = [], []
    save_idx = -1
    dataset_len = len(dataloader)
    with torch.no_grad():
        for batch_idx, (batch, index) in enumerate(dataloader):
            # index: (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, episode_end)
            episode_len = index[4][0].item()
            episode_cnt = index[5].tolist()
            batch['obs']['image'] = batch['obs']['image'][:2, ...]
            if batch['obs']['image'].size(0)  < xlstm_cfg.batchsize:
                batch['obs']['image'] = pad_to_batch_size([batch['obs']['image']], xlstm_cfg.batchsize)[0]
                batch['obs']['agent_pos'] = pad_to_batch_size([batch['obs']['agent_pos']], xlstm_cfg.batchsize)[0]
                batch['action'] = pad_to_batch_size([batch['action']], xlstm_cfg.batchsize)[0]
            batch['obs'] = linear_norm.normalize(batch['obs'])
            batch['action'] = linear_norm['action'].normalize(batch['action'])
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_label = batch['obs']['image'][:, xlstm_cfg.past_img_num:, ...]
            output, run_time = inference_time(model, batch, phase)
            loss_results = compute_losses(output, batch, writer, epc, valid_avg, phase)
            output.actions = linear_norm['action'].unnormalize(output.actions)
            loss_results.new_actions = linear_norm['action'].unnormalize(loss_results.new_actions)
            batch['action'] = linear_norm['action'].unnormalize(batch['action'])
            valid_avg.infer_time.update(run_time)
            
            if val_indices_list[epc%len(val_indices_list)] in episode_cnt:
                if not xlstm_cfg.olny_action_generate:
                    grip_results =  compute_ssimpsnr.compute(output.obs_future_seq, obs_label)
                    update_ssim_psnr(grip_results, valid_avg)
                    pred_grip_images.append(output.obs_future_seq_more)
                    label_grip_images.append(batch['obs']['image'])
                old_pred_action.append(output.actions)
                pred_action.append(loss_results.new_actions)
                label_action.append(batch['action'])
            print(f"\tPhase: valid | Epc: {epc}/{xlstm_cfg.epochs} | Total: {batch_idx}/{len(dataloader)} | losses: {loss_results.losses.item():.2f} | inference_time: {run_time:.4f}", end='\r')
            # if batch_idx > 1 and epc <= 115:
            #     break

        if len(pred_action) == 1:
            pred_actions = pred_action[-1]
            label_actions = label_action[-1]
            old_pred_actions = old_pred_action[-1]
            if not xlstm_cfg.olny_action_generate:
                pred_grip_images = pred_grip_images[-1]
                label_grip_images = label_grip_images[-1]
        elif len(pred_action) > 1: 
            pred_actions = torch.cat(pred_action, dim=0)
            label_actions = torch.cat(label_action, dim=0)
            old_pred_actions = torch.cat(old_pred_action, dim=0)
            if not xlstm_cfg.olny_action_generate:
                pred_grip_images = torch.cat(pred_grip_images, dim=0)
                label_grip_images = torch.cat(label_grip_images, dim=0)

        pred_total_act_dict = tens2act(pred_actions, epc, name='pred_total_new_act')
        label_total_act_dict = tens2act(label_actions, epc, name='label_total_act')
        pred_total_old_act_dict = tens2act(old_pred_actions, epc, name='pred_total_old_act')
        best_results.total_actions = {**pred_total_act_dict, **label_total_act_dict}
        best_results.old_actions = {**pred_total_old_act_dict, **label_total_act_dict}

        if pred_grip_images is not None and not xlstm_cfg.olny_action_generate:
            best_results.grip_upsample_frame_pred = pred_grip_images
            best_results.grip_frame_labels = label_grip_images
        else:
            best_results.grip_upsample_frame_pred = None
            best_results.grip_frame_labels = None

    return best_results

from typing import Dict, Callable, List
def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def init_weights(m):
    # 初始化所有线性层和卷积层（1D/2D/3D）
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # 使用 Kaiming 初始化（假设使用 ReLU 激活）
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置为 0（如果存在）
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    # 初始化所有 BatchNorm 层（1D/2D/3D）
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)     # gamma 初始化为 1
        nn.init.zeros_(m.bias)      # beta 初始化为 0

def run_time(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_ = time.time() - start_time
        return result, time_
    return wrapper

@run_time
def inference_time(model, *args, **kwargs):
    
    return model(*args, **kwargs)


if __name__ == "__main__":
    # remarks that  when training on windows
    # if not rospy.core.is_initialized():
    #     rospy.init_node('act_generate', anonymous=True, log_level=rospy.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--config", default=r"./pusht_config.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    xlstm_cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(xlstm_cfg)
    main(xlstm_cfg)



 






