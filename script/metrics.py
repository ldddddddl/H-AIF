import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import butter, filtfilt

def compute_jerk_metrics(traj, dt):
    """
    计算三阶导数（jerk）相关的平滑度指标。
    
    Args:
        traj: ndarray, shape (N, D)，D为维度（如9维）。
        dt: float，时域采样间隔。
    
    Returns:
        mean_sq_jerk: float，均方加加速度值；
        normalized_jerk: float，归一化加加速度值。
    """
    # 速度 v = d x / dt
    vel = np.gradient(traj, dt, axis=0)
    # 加速度 a = d v / dt
    acc = np.gradient(vel, dt, axis=0)
    # 加加速度 jerk = d a / dt
    jerk = np.gradient(acc, dt, axis=0)

    # 轨迹总时长
    T = traj.shape[0] * dt

    # 每个时刻的 jerk 平方和
    jerk_sq = np.sum(jerk**2, axis=1)

    # 均方加加速度
    mean_sq_jerk = np.mean(jerk_sq)

    # 归一化加加速度
    normalized_jerk = 0.5 * (T**5) * np.mean(jerk_sq)

    return mean_sq_jerk, normalized_jerk


def count_accel_zero_crossings(traj, dt, cutoff_ratio=0.1):
    """
    统计加速度模长信号的零交叉次数（抖动指标之一）。
    
    Args:
        traj: ndarray, shape (N, D)；
        dt: float，时域采样间隔；
        cutoff_ratio: float，低通滤波截止频率 / Nyquist 比例（可选）。
    
    Returns:
        zc_count: int，零交叉次数。
    """
    # 计算加速度
    vel = np.gradient(traj, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    # 取加速度模长
    acc_mag = np.linalg.norm(acc, axis=1)

    # 可选低通滤波，减少高频噪声影响
    nyq = 0.5 / dt
    cutoff = cutoff_ratio * nyq
    b, a = butter(2, cutoff / nyq, fs=1/dt)
    acc_filt = filtfilt(b, a, acc_mag)

    # 零交叉：信号正负变化的点
    zc_indices = np.where(np.diff(np.sign(acc_filt)) != 0)[0]
    return len(zc_indices)

def compute_path_inefficiency(traj):
    """
    归一化路径效率 PI = (TV - L_direct) / L_direct
    Args:
        traj: ndarray, shape (N, D)
    Returns:
        PI: float
    """
    # TV: total variation
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    TV = diffs.sum()
    # L_direct: 端到端直线距离
    L_direct = np.linalg.norm(traj[-1] - traj[0])
    PI = (TV - L_direct) / L_direct
    return PI

def compute_highfreq_ratio(traj, dt, fc_ratio=0.2):
    """
    高频能量占比 R_HF
    Args:
        traj: ndarray, shape (N, D)
        dt: float, 采样间隔
        fc_ratio: float, 截止频率 / Nyquist 比例
    Returns:
        R_HF: float
    """
    # 速度大小
    vel = np.gradient(traj, dt, axis=0)
    vmag = np.linalg.norm(vel, axis=1)
    # FFT
    N = vmag.shape[0]
    V = np.fft.rfft(vmag)
    P = np.abs(V)**2
    # 频率轴
    nyq = 0.5 / dt
    freqs = np.fft.rfftfreq(N, d=dt)
    # 高频部分
    idx_cut = freqs > (fc_ratio * nyq)
    E_hf = P[idx_cut].sum()
    E_total = P.sum()
    return E_hf / E_total

def compute_acc_energy(traj, dt):
    """
    加速度能量 E_acc = sum ||a||^2 * dt
    Args:
        traj: ndarray, shape (N, D)
        dt: float, 采样间隔
    Returns:
        E_acc: float
    """
    vel = np.gradient(traj, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    # 每个时刻的加速度平方和
    acc_sq = np.sum(acc**2, axis=1)
    return acc_sq.sum() * dt


def calc_metrics(inputs, avg, flag:str=None):
    inputs = inputs.cpu().detach()
    dt = 0.001  # 采样间隔，1ms
    msj, nj = compute_jerk_metrics(inputs, dt)
    zc = count_accel_zero_crossings(inputs, dt, cutoff_ratio=0.1)
    pi = compute_path_inefficiency(inputs)
    hf = compute_highfreq_ratio(inputs, dt, fc_ratio=0.2)
    ea = compute_acc_energy(inputs, dt)
    if flag == 'act':
        avg.act_jerk_mean.update(msj)
        avg.act_jerk_nrom.update(nj)
        avg.act_azc.update(zc)
        avg.act_pi.update(pi)
        avg.act_hf.update(hf)
        avg.act_ea.update(ea)
    elif flag == 'new_act':
        avg.new_act_jerk_mean.update(msj)
        avg.new_act_jerk_nrom.update(nj)
        avg.new_act_azc.update(zc)
        avg.new_act_pi.update(pi)
        avg.new_act_hf.update(hf)
        avg.new_act_ea.update(ea)
    elif flag == 'label':
        avg.label_act_jerk_mean.update(msj)
        avg.label_act_jerk_nrom.update(nj)
        avg.label_act_azc.update(zc)
        avg.label_act_pi.update(pi)
        avg.label_act_hf.update(hf)
        avg.label_act_ea.update(ea)
    
    
# ==== 示例用法 ====
if __name__ == "__main__":
    # 假设有一个 [1600, 9] 的轨迹数据
    N, D = 1600, 9
    # 用随机游走数据做示例，真实请替换为你的 traj 数组
    traj = np.cumsum(0.01 * np.random.randn(N, D), axis=0)
    dt = 0.001  # 采样间隔，1ms

    # 计算 jerk 指标
    msj, nj = compute_jerk_metrics(traj, dt)
    print(f"Mean squared jerk: {msj:.4e}")
    print(f"Normalized jerk:   {nj:.4e}")

    # 计算加速度零交叉次数
    zc = count_accel_zero_crossings(traj, dt, cutoff_ratio=0.1)
    print(f"Acceleration zero crossings: {zc}")


    pi = compute_path_inefficiency(traj)
    hf = compute_highfreq_ratio(traj, dt, fc_ratio=0.2)
    ea = compute_acc_energy(traj, dt)

    print(f"Path Inefficiency:      {pi:.4f}")
    print(f"High-Freq Energy Ratio: {hf:.4f}")
    print(f"Acceleration Energy:    {ea:.4e}")