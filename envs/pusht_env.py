import gymnasium as gym
import gym_pusht
import torch
from torch import optim
from torch.nn import functional as F
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
import numpy as np
import queue
import copy
import cv2
import time
import os
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from model.pustht_models import ActNet
from pusht_losses import ComputeLosses
from pusht_misc import InitAverager
from script.normalizer import LinearNormalizer
from pusht_image_dataset import PushTImageDataset
from pusht_base_dataset import BaseImageDataset





DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
IS_USE_CUDA = torch.cuda.is_available()



def load_model(config):

    model = ActNet(config, IS_USE_CUDA, DEVICE).to(DEVICE)
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=xlstm_cfg.lr,
        betas=[0.9, 0.99],
        weight_decay = 1E-4
        )
    linear_norm = LinearNormalizer().to(DEVICE)
    # configure dataset
    dataset: BaseImageDataset
    dataset = PushTImageDataset(**xlstm_cfg.task.dataset)
    # assert isinstance(dataset, BaseImageDataset)
    normalizer = dataset.get_normalizer()
    linear_norm.load_state_dict(normalizer.state_dict())

    compute_loss = ComputeLosses(DEVICE, config).to(DEVICE)
    model_path = os.path.join(config.checkpoint_path, 'check_point')
    model_name = os.listdir(model_path)[-1]
    
    checkpoint = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer, model, compute_loss, linear_norm

def workspace(model: ActNet=None, optimizer: optim=None, compute_loss: ComputeLosses=None, linear_norm=None, config=None):
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    observation, info = env.reset()
    batch = {
        'obs':{
            'image':[],
            'agent_pos':[]
            },
        'action':[]
    }
    valid_avg = InitAverager()
    action = env.action_space.sample()
    img_list, agent_pos_list, action_list = [], [], []
    for step in range(1000):
        if step == 0 or config.envs.mode == 'single_step':
            img = env.render()
            img_list.append(img)
            agent_pos = observation[:2]
            agent_pos_list.append(agent_pos)
            action_list.append(action)
            cv2.imshow("PushT", img[:, :, ::-1])  # 转成 BGR 显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        tbatch, batch = process_data(img_list, agent_pos_list, action_list, step, batch, config)
        img_list, agent_pos_list, action_list = [], [], []
        tbatch = dict_apply(tbatch, lambda x: x.to(DEVICE, non_blocking=True))
        tbatch = dict_apply(tbatch, lambda x: x.to(torch.float32))
        output = model(tbatch)
        loss_results = compute_loss(output, tbatch, writer=None, epoch=step, avg_=valid_avg)
        loss_results.new_actions = linear_norm['action'].unnormalize(loss_results.new_actions)
        action = np.asarray(loss_results.new_actions[-1].cpu().detach())
        if config.envs.mode == 'series':
            for a in action:
                observation, reward, terminated, truncated, info = env.step(a)
                img = env.render()
                agent_pos = observation[:2]
                agent_pos_list.append(agent_pos)
                img_list.append(img)
                action_list.append(a)
                cv2.imshow("PushT", img[:, :, ::-1])  # 转成 BGR 显示
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.3)
        else: # single step
            action_list.append(action[0])
            observation, reward, terminated, truncated, info = env.step(action[0])
            
        print(f'current step:{step} | current (x, y):({action[0]}, {action[1]})')
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    cv2.destroyAllWindows()
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

def process_data(image, state, action, step, batch, config):
    for i, s, a in zip(image, state, action):
        nimage = process_image(i)
        nimages = np.stack(nimage, axis=0)
        nimages = torch.from_numpy(nimages).unsqueeze(0)
        re_img = F.interpolate(nimages, size=(96, 96))
        re_img = re_img.repeat(config.batchsize, 1, 1, 1, 1)
        
        agent_positions = np.stack(s, 0)  # (T, 2)
        agent_positions = torch.from_numpy(agent_positions).unsqueeze(0)
        agent_positions = agent_positions.repeat(config.batchsize, 1, 1)
        
        actions = np.stack(a, 0).astype(np.float32)  # (T, 2)
        actions = torch.from_numpy(actions).unsqueeze(0)
        actions = actions.repeat(config.batchsize, 1, 1)

        tbatch, batch = add2queue(re_img, agent_positions, actions, batch, step, config)
    return tbatch, batch

def add2queue(image, state, action, batch, step, config):
    tbatch = copy.deepcopy(batch)
    if step == 0:
        for _ in range(config.past_img_num*config.per_image_with_signal_num):
            batch['obs']['image'].append(image)
            batch['obs']['agent_pos'].append(state)
            batch['action'].append(action)
    else:
        batch['obs']['image'].append(image)
        batch['obs']['agent_pos'].append(state)
        batch['action'].append(action)
        batch['obs']['image'].pop()
        batch['obs']['agent_pos'].pop()
        batch['action'].pop()
    tbatch['obs']['image'] = torch.cat(batch['obs']['image'], dim=1)
    tbatch['obs']['agent_pos'] = torch.cat(batch['obs']['agent_pos'], dim=1)
    tbatch['action'] = torch.cat(batch['action'], dim=1)
    
    return tbatch, batch

def process_image(raw_img):
    # 1. 调整通道顺序
    img = np.transpose(raw_img, (2,0,1))  # HWC -> CHW
    
    # 2. 归一化到 [0,1]
    img = img.astype(np.float32) / 255.0

    return img

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default=r"pusht_config.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    xlstm_cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(xlstm_cfg)
    optimizer, model, compute_loss, linear_norm = load_model(xlstm_cfg)
    workspace(model, optimizer, compute_loss, linear_norm, xlstm_cfg)
    # workspace(config=xlstm_cfg)
