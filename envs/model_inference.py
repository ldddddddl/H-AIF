import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('.')
from model.pustht_models import ActNet
from pusht_misc import InitAverager
from pusht_losses import ComputeLosses

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_image(raw_img):
    img = np.transpose(raw_img, (2,0,1)).astype(np.float32) / 255.0
    return torch.from_numpy(img)

def run_inference(cfg):
    # 加载配置和模型

    model = ActNet(cfg, torch.cuda.is_available(), device).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 初始化环境
    env = gym.make("gym_pusht/PushT-v0", 
                  render_mode="rgb_array",
                  observation_width=96,
                  observation_height=96)
    obs, _ = env.reset()
    action = env.action_space.sample()

    # 初始化数据缓冲区
    batch = {
        'obs': {
            'image': [],
            'agent_pos': []
        },
        'action': []
    }
    valid_avg = InitAverager()

    for step in range(1000):
        # 获取观测数据
        img = env.render()
        agent_pos = obs[:2]

        # 图像预处理
        processed_img = process_image(img)
        processed_img = F.interpolate(processed_img.unsqueeze(0), size=(96, 96))
        
        # 构建时序数据（维护最近10个时间步）
        if len(batch['obs']['image']) >= 10:
            for k in batch['obs']:
                batch['obs'][k].pop(0)

        # 添加当前步数据
        batch['obs']['image'].append(processed_img.unsqueeze(0))  # [1,3,96,96]
        batch['obs']['agent_pos'].append(torch.from_numpy(agent_pos).float().unsqueeze(0))  # [1,2]
        batch['action'].append(torch.from_numpy(action).float().unsqueeze(0))  # [1,2]
        # 构建模型输入（重复8次形成batch）
        model_input = {
            'obs': {
                'image': torch.cat(batch['obs']['image'][-10:]).repeat(8,10,1,1,1).permute(0,1,2,3,4),  # [8,10,3,96,96]
                'agent_pos': torch.cat(batch['obs']['agent_pos'][-10:]).repeat(8,10,1)  # [8,10,2]
            },
            'action': torch.cat(batch['action'][-10:]).repeat(8,10,1),  # [8,10,2]
        }
        
        # 模型推理
        with torch.no_grad():
            output = model(model_input)
            loss_results = compute_loss(output, tbatch, writer=None, epoch=step, avg_=valid_avg)
            action_seq = loss_results.new_actions
        # 根据配置选择执行模式
        if cfg.envs.mode == 'series':
            for a in action_seq[0]:  # 执行预测的10个动作
                obs, _, terminated, truncated, _ = env.step(a.cpu().numpy())
                if terminated or truncated:
                    break
        else:  # 单步模式
            action = action_seq[0,-1].cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
        obs, _, terminated, truncated, _ = env.step(action)

        # 环境重置
        if terminated or truncated:
            obs, _ = env.reset()

        # 可视化
        cv2.imshow('PushT', img[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default=r"pusht_config.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    xlstm_cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(xlstm_cfg)
    run_inference(xlstm_cfg)