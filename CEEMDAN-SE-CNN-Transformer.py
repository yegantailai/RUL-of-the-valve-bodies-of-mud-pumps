# ============================================================
# v30-最终稳定版：PyEMD CPU极速降采样 + 断点续传
# ============================================================

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.stats import kurtosis, skew, pearsonr
from scipy.signal import welch, find_peaks, hilbert, savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter, gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt

# ★ 恢复使用最稳定的 PyEMD
try:
    from PyEMD import CEEMDAN
except ImportError:
    print("请先安装: pip install EMD-signal")
    raise

matplotlib.rc("font", family='Microsoft YaHei')
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
_candidates = [
    r'G:\慧油科技数据\寿命预测', r'I:\慧油科技数据\寿命预测',
    r'D:\慧油科技数据\寿命预测', r'E:\慧油科技数据\寿命预测',
]
BASE_DIR = None
for _p in _candidates:
    if os.path.exists(os.path.join(_p, '7205')):
        BASE_DIR = _p
        break

RESULT_DIR = os.path.join(BASE_DIR, 'deliverable')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

FS_ORIGINAL = 20000
DOWNSAMPLE_FACTOR = 10  # 降采样 10 倍
FS = FS_ORIGINAL // DOWNSAMPLE_FACTOR

CHUNK_SIZE = 2048
CHUNK_STEP = 20000  # 步长拉大，极大减少计算次数
WINDOW_SIZE = 64
STEP = 4
BATCH_SIZE = 64
EPOCHS = 40
LEARN_RATE = 0.0003
FEATURE_SMOOTH_WIN = 15
MONO_BETA = 0.20
SMOOTH_GAMMA = 0.08
ASYM_PENALTY = 0.30
TOP_K_FEATURES = 15
POST_MEDIAN_WIN = 51
POST_GAUSS_SIGMA = 80
MC_SAMPLES = 20
END_RUL_RESIDUAL = 0.0

torch.manual_seed(100)
np.random.seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================================
# 1. 核心函数 (PyEMD CPU 版)
# ============================================================
def process_block_with_ceemdan(block):
    """使用 PyEMD 的 CEEMDAN 降噪，trials 降为 3 提速"""
    try:
        if len(block) < 100: return block
        ceemdan = CEEMDAN(trials=3)
        IMFs = ceemdan(block)
        correlations = []
        for imf in IMFs:
            corr, _ = pearsonr(imf, block) if np.std(imf) > 0 else (0.0, None)
            correlations.append(abs(corr))
        top3_idx = np.argsort(correlations)[-3:]
        return np.sum(IMFs[top3_idx], axis=0)
    except Exception as e:
        return block


def higuchi_fd(x, kmax=3):
    x = np.array(x, dtype=float)
    N = len(x)
    if N < 20: return 0.0
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2: continue
            Lm = np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / ((len(idx) - 1) * k)
            Lk.append(Lm)
        if Lk: L.append(np.mean(Lk))
    if len(L) < 2: return 0.0
    L = np.array(L)
    try:
        return np.polyfit(np.log(1.0 / np.arange(1, len(L) + 1)), np.log(L + 1e-12), 1)[0]
    except:
        return 0.0


def extract_hht_features(signal):
    try:
        envelope = np.abs(hilbert(signal))
        N = len(envelope)
        env_fft = np.abs(rfft(envelope)) / N
        env_freqs = rfftfreq(N, d=1.0 / FS)
        half = N // 4
        env_fft_half, env_freqs_half = env_fft[:half], env_freqs[:half]
        s = np.sum(env_fft_half)
        if s > 0:
            c = np.sum(env_freqs_half * env_fft_half) / s
            pf = env_freqs_half[np.argmax(env_fft_half)]
            pa = np.max(env_fft_half)
            p = env_fft_half / s;
            p = p[p > 0]
            e = -np.sum(p * np.log(p))
        else:
            c, pf, pa, e = 0.0, 0.0, 0.0, 0.0
        return {'HHT_EnvCentroid': c, 'HHT_EnvPeakFreq': pf, 'HHT_EnvPeakAmp': pa, 'HHT_EnvEntropy': e}
    except:
        return {'HHT_EnvCentroid': 0.0, 'HHT_EnvPeakFreq': 0.0, 'HHT_EnvPeakAmp': 0.0, 'HHT_EnvEntropy': 0.0}


def extract_features_enhanced(signal, signal_denoised):
    signal = np.asarray(signal, dtype=float);
    signal_denoised = np.asarray(signal_denoised, dtype=float)
    signal = signal[np.isfinite(signal)];
    signal_denoised = signal_denoised[np.isfinite(signal_denoised)]
    if len(signal) < 20 or len(signal_denoised) < 20: return None
    kurt = kurtosis(signal_denoised)
    if not np.isfinite(kurt): kurt = 0.0
    rms = np.sqrt(np.mean(signal_denoised ** 2));
    peak = np.max(np.abs(signal_denoised))
    peak_factor = peak / (rms + 1e-12);
    pulse_factor = peak / (np.mean(np.abs(signal_denoised)) + 1e-12)
    skewness = skew(signal_denoised)
    if not np.isfinite(skewness): skewness = 0.0
    hist, _ = np.histogram(signal, bins=50, density=True)
    p = hist[hist > 0];
    ent = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
    fd = higuchi_fd(signal)
    N = len(signal);
    fft_vals = rfft(signal);
    fft_abs = np.abs(fft_vals) / N;
    fft_freqs = rfftfreq(N, d=1.0 / FS)
    sum_abs = np.sum(fft_abs)
    if sum_abs > 0:
        centroid = np.sum(fft_freqs * fft_abs) / sum_abs
        msf = np.sum((fft_freqs ** 2) * fft_abs) / sum_abs
        varf = np.sum(((fft_freqs - centroid) ** 2) * fft_abs) / sum_abs
        main_freq_idx = np.argmax(fft_abs);
        main_freq_amp = fft_abs[main_freq_idx];
        main_freq = fft_freqs[main_freq_idx]
    else:
        centroid, msf, varf, main_freq_amp, main_freq = 0.0, 0.0, 0.0, 0.0, 0.0
    nperseg = min(N, 1024);
    freq, ps = welch(signal, fs=FS, nperseg=nperseg)
    ps = np.maximum(ps, 1e-12);
    total_energy = np.sum(ps)
    spectral_flatness = np.exp(np.mean(np.log(ps))) / (np.mean(ps) + 1e-12)
    band_energies = [];
    band_edges = np.linspace(0, len(ps), 6, dtype=int)
    for i in range(5):
        band_energy = np.sum(ps[band_edges[i]:band_edges[i + 1]]);
        band_energies.append(band_energy / (total_energy + 1e-12))
    try:
        envelope = np.abs(hilbert(signal));
        env_freq, env_ps = welch(envelope, fs=FS, nperseg=nperseg)
        env_peaks, _ = find_peaks(env_ps, height=np.mean(env_ps))
        env_peak_freq = env_freq[env_peaks[np.argmax(env_ps[env_peaks])]] if len(env_peaks) > 0 else 0.0
    except:
        env_peak_freq = 0.0
    feat = {
        'Kurtosis': kurt, 'Entropy': ent, 'Fractal_Dimension': fd, 'Peak_factor': peak_factor,
        'Pulse_factor': pulse_factor,
        'Skewness': skewness, 'Rms_vibration': rms, 'Peak_vibration': peak, 'FFT_Centroid': centroid, 'FFT_MSF': msf,
        'FFT_VarF': varf, 'FFT_Main_Freq': main_freq, 'FFT_Main_Amp': main_freq_amp,
        'Spectral_flatness': spectral_flatness,
        'Band1_E': band_energies[0], 'Band2_E': band_energies[1], 'Band3_E': band_energies[2],
        'Band4_E': band_energies[3],
        'Band5_E': band_energies[4], 'Env_Peak_Freq': env_peak_freq
    }
    feat.update(extract_hht_features(signal))
    return feat


DUMMY_SIGNAL = np.zeros(1024)
FEATURE_NAMES = list(extract_features_enhanced(DUMMY_SIGNAL, DUMMY_SIGNAL).keys())


# ============================================================
# 2. 流式读取 (带断点续传 + 降采样)
# ============================================================
def process_file_streaming(path, unit_name):
    cache_path = os.path.join(RESULT_DIR, f'cache_{unit_name}.parquet')
    if os.path.exists(cache_path):
        print(f"  [断点续传] 发现已有缓存，直接秒读单元: {unit_name}")
        return cache_path

    print(f"  流式处理单元: {unit_name} (CPU狂算中，已降采样提速) ...")
    feats_list = []
    rms_list = []

    chunk_iter = pd.read_csv(path, header=None, usecols=[0], dtype=np.float32, chunksize=100000)

    for chunk in chunk_iter:
        # ★ 降采样
        sig_chunk = chunk.values.flatten()[::DOWNSAMPLE_FACTOR]

        sig_denoised = process_block_with_ceemdan(sig_chunk)

        for i in range(0, len(sig_chunk) - CHUNK_SIZE + 1, CHUNK_STEP):
            ft = extract_features_enhanced(sig_chunk[i:i + CHUNK_SIZE], sig_denoised[i:i + CHUNK_SIZE])
            if ft:
                feats_list.append(ft)
                rms_list.append(np.sqrt(np.mean(sig_denoised[i:i + CHUNK_SIZE] ** 2)))

        del sig_chunk, sig_denoised, chunk
        gc.collect()

    df = pd.DataFrame(feats_list)
    df.columns = FEATURE_NAMES
    rms_seq = np.array(rms_list)

    window = min(101, len(rms_seq) - 1) if len(rms_seq) % 2 == 1 else min(100, len(rms_seq) - 2)
    if window % 2 == 0: window += 1
    smoothed_rms = savgol_filter(rms_seq, window_length=window, polyorder=3, mode='nearest')
    n = len(smoothed_rms)
    healthy_len = max(int(n * 0.1), 10)
    base_data = smoothed_rms[:healthy_len]
    mu = np.mean(base_data);
    sigma = np.std(base_data, ddof=1)
    threshold = mu + 3 * sigma
    fpt_idx = n - 1
    for i in range(healthy_len, n):
        if smoothed_rms[i] >= threshold:
            fpt_idx = i
            break

    print(f"    [FPT 检测] 退化点: {fpt_idx} (总长: {len(df)})")
    rul = np.ones(n)
    if n - fpt_idx > 1:
        t = np.linspace(0, 1, n - fpt_idx)
        rul[fpt_idx:] = 1.0 - t
    df['rul'] = rul
    df['unit'] = unit_name

    df.to_parquet(cache_path)
    print(f"    [保存成功] 单元 {unit_name} 的特征已存盘！")
    return cache_path


# ============================================================
# 3-5. 模型、训练与预测
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid())

    def forward(self, x):
        b, c, t = x.size();
        return x * self.fc(x.mean(dim=2)).view(b, c, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model);
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


class CNNTransformerSeq(nn.Module):
    def __init__(self, input_dim, conv_channels=32, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1), nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2), nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True))
        self.se = SEBlock(conv_channels, reduction=8)
        self.pos_enc = PositionalEncoding(conv_channels, max_len=WINDOW_SIZE + 4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_channels, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head_rul = nn.Linear(conv_channels, 1);
        self.head_hi = nn.Linear(conv_channels, 1)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1));
        x = self.se(x);
        x = self.pos_enc(x.permute(0, 2, 1));
        x = self.transformer(x)
        return self.head_rul(x).squeeze(-1), self.head_hi(x).squeeze(-1)


class SequenceRULLoss(nn.Module):
    def __init__(self, beta=0.20, gamma=0.08, hi_weight=0.3, asym_penalty=0.3):
        super().__init__()
        self.beta, self.gamma, self.hi_weight, self.asym_penalty = beta, gamma, hi_weight, asym_penalty
        self.huber = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_rul, pred_hi, target):
        weights = 1.0 + 2.0 * target
        loss_main = (self.huber(pred_rul, target) * weights).mean()
        diff = pred_rul[:, 1:] - pred_rul[:, :-1];
        mono = torch.relu(diff).mean()
        smooth = (pred_rul[:, 2:] - 2 * pred_rul[:, 1:-1] + pred_rul[:, :-2]).pow(2).mean() if pred_rul.size(
            1) >= 3 else torch.tensor(0.0, device=pred_rul.device)
        loss_hi = self.huber(pred_hi, target).mean()
        over_est = torch.relu(pred_rul - target);
        loss_asym = (over_est * weights).mean() * self.asym_penalty
        return loss_main + self.beta * mono + self.gamma * smooth + self.hi_weight * loss_hi + loss_asym


def predict_unit_full_mc(model, sub_df, feature_cols, window_size, step, device, batch_sz=128, n_samples=20):
    data = sub_df[feature_cols].values.astype(np.float32)
    n = len(data)
    if n < window_size: return np.full(n, np.nan), np.full(n, np.nan)
    windows, starts = [], []
    for i in range(0, n - window_size + 1, step):
        windows.append(data[i:i + window_size]);
        starts.append(i)
    windows = torch.tensor(np.array(windows), dtype=torch.float32)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d): m.eval()
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            sample_preds = []
            for b in range(0, len(windows), batch_sz):
                sample_preds.append(model(windows[b:b + batch_sz].to(device))[0].cpu().numpy())
            all_preds.append(np.concatenate(sample_preds, axis=0))
    all_preds = np.stack(all_preds, axis=0)
    preds_sum, preds_std_sum, counts = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, i in enumerate(starts):
        for j in range(window_size):
            preds_sum[i + j] += all_preds.mean(axis=0)[k, j]
            preds_std_sum[i + j] += all_preds.std(axis=0)[k, j] ** 2
            counts[i + j] += 1
    preds_mean, preds_std = np.full(n, np.nan), np.full(n, np.nan)
    valid = counts > 0
    preds_mean[valid] = preds_sum[valid] / counts[valid]
    preds_std[valid] = np.sqrt(preds_std_sum[valid] / counts[valid])
    return preds_mean, preds_std


if __name__ == '__main__':
    try:
        import multiprocessing;

        multiprocessing.freeze_support()
    except:
        pass

    sensor_configs = [
        (os.path.join(BASE_DIR, '7201', '数据处理', '合并'), '7201_sensor2', '7201_s2', True),
        (os.path.join(BASE_DIR, '7201', '数据处理', '合并'), '7201_sensor4', '7201_s4', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor3', '7205_P1_s3', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor4', '7205_P1_s4', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor5', '7205_P1_s5', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor6', '7205_P1_s6', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor1', '7205_P2_s1', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor2', '7205_P2_s2', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor3', '7205_P2_s3', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor4', '7205_P2_s4', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor5', '7205_P2_s5', True),
        (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor6', '7205_P2_s6', False),
    ]

    print("开始流式提取特征并存盘 (CPU极速降采样)...")
    train_cache_paths, test_cache_paths = [], []
    for folder, sensor_name, unit_tag, is_train in sensor_configs:
        path = os.path.join(folder, f"{sensor_name}.csv")
        if os.path.exists(path):
            cache_path = process_file_streaming(path, unit_tag)
            if is_train:
                train_cache_paths.append(cache_path)
            else:
                test_cache_paths.append(cache_path)

    print("\n所有特征已存盘，开始加载并合并特征...")
    train_df = pd.concat([pd.read_parquet(p) for p in train_cache_paths], ignore_index=True)
    test_df = pd.concat([pd.read_parquet(p) for p in test_cache_paths], ignore_index=True)
    all_feature_cols = FEATURE_NAMES


    def smooth_features_inplace(df, feature_cols, window=15):
        for unit in df['unit'].unique():
            mask = df['unit'] == unit
            df.loc[mask, feature_cols] = df.loc[mask, feature_cols].rolling(window=window, min_periods=1,
                                                                            center=True).mean().values
        return df


    train_df = smooth_features_inplace(train_df, all_feature_cols)
    test_df = smooth_features_inplace(test_df, all_feature_cols)

    mi_scores = mutual_info_regression(train_df[all_feature_cols].values, train_df['rul'].values, random_state=100)
    mi_df = pd.DataFrame({'feature': all_feature_cols, 'mi': mi_scores}).sort_values('mi', ascending=False).reset_index(
        drop=True)
    feature_cols = mi_df['feature'].head(TOP_K_FEATURES).tolist()

    for df in [train_df, test_df]:
        for unit in df['unit'].unique():
            mask = df['unit'] == unit
            data = df.loc[mask, feature_cols].values
            local_scaler = StandardScaler()
            df.loc[mask, feature_cols] = local_scaler.fit_transform(data)

    scaler_y = MinMaxScaler()
    train_df[['rul']] = scaler_y.fit_transform(train_df[['rul']])
    test_df[['rul']] = scaler_y.transform(test_df[['rul']])


    def make_windows_per_unit(df, feature_cols, window_size, step=1):
        X_list, y_list = [], []
        for unit in df['unit'].unique():
            sub = df[df['unit'] == unit].reset_index(drop=True)
            if len(sub) <= window_size: continue
            data = sub[feature_cols].values.astype(np.float32)
            labels = sub['rul'].values.astype(np.float32)
            for i in range(0, len(sub) - window_size + 1, step):
                X_list.append(data[i:i + window_size])
                y_list.append(labels[i:i + window_size])
        return np.array(X_list), np.array(y_list)


    train_X, train_y = make_windows_per_unit(train_df, feature_cols, WINDOW_SIZE, STEP)
    test_X, test_y = make_windows_per_unit(test_df, feature_cols, WINDOW_SIZE, STEP)

    del train_df, test_df;
    gc.collect()

    train_set = torch.tensor(train_X, dtype=torch.float32)
    train_label = torch.tensor(train_y, dtype=torch.float32)
    test_set = torch.tensor(test_X, dtype=torch.float32)
    test_label = torch.tensor(test_y, dtype=torch.float32)

    train_loader = Data.DataLoader(Data.TensorDataset(train_set, train_label), batch_size=BATCH_SIZE, shuffle=True,
                                   drop_last=True)
    test_loader = Data.DataLoader(Data.TensorDataset(test_set, test_label), batch_size=BATCH_SIZE, shuffle=False,
                                  drop_last=False)

    model = CNNTransformerSeq(train_set.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARN_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_function = SequenceRULLoss(beta=MONO_BETA, gamma=SMOOTH_GAMMA, hi_weight=0.3, asym_penalty=ASYM_PENALTY)

    print('\n开始训练...')
    best_loss = float('inf')
    best_model_path = os.path.join(RESULT_DIR, 'best_model.pt')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device);
            optimizer.zero_grad()
            pred_rul, pred_hi = model(seq);
            loss = loss_function(pred_rul, pred_hi, labels)
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0);
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        val_loss = []
        with torch.no_grad():
            for seq, labels in test_loader:
                seq, labels = seq.to(device), labels.to(device)
                pred_rul, pred_hi = model(seq)
                val_loss.append(loss_function(pred_rul, pred_hi, labels).item())
        avg_val = np.average(val_loss);
        scheduler.step(avg_val)
        if avg_val < best_loss:
            best_loss = avg_val;
            torch.save(model.state_dict(), best_model_path)
            flag = " ← 保存最佳模型"
        else:
            flag = ""
        print(f'Epoch {epoch + 1:2} | Train: {np.average(train_loss):.4f} | Val: {avg_val:.4f}{flag}')

    test_df_eval = pd.read_parquet(test_cache_paths[0])
    for unit in test_df_eval['unit'].unique():
        mask = test_df_eval['unit'] == unit
        data = test_df_eval.loc[mask, feature_cols].values
        local_scaler = StandardScaler()
        test_df_eval.loc[mask, feature_cols] = local_scaler.fit_transform(data)

    test_df_eval['rul_real'] = test_df_eval['rul']

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_df_eval['pred_raw'] = np.nan
    test_df_eval['pred_post'] = np.nan

    for unit in test_df_eval['unit'].unique():
        mask = test_df_eval['unit'] == unit
        sub = test_df_eval.loc[mask].reset_index()
        preds, _ = predict_unit_full_mc(model, sub, feature_cols, WINDOW_SIZE, STEP, device)
        valid = ~np.isnan(preds)
        inv = np.full_like(preds, np.nan);
        inv[valid] = preds[valid]

        y = inv[valid]
        healthy_len = max(int(len(y) * 0.1), 5)
        base_val = np.median(y[:healthy_len])
        y_calibrated = y - base_val + 1.0
        y_calibrated = np.clip(y_calibrated, 0.0, 1.0)
        y_calibrated = gaussian_filter1d(median_filter(y_calibrated, size=POST_MEDIAN_WIN), sigma=POST_GAUSS_SIGMA)
        y_calibrated = np.minimum.accumulate(y_calibrated)

        post = inv.copy();
        post[valid] = y_calibrated
        test_df_eval.loc[mask, 'pred_raw'] = inv
        test_df_eval.loc[mask, 'pred_post'] = post

    valid = test_df_eval.dropna(subset=['pred_post', 'rul_real'])
    print('\n' + '=' * 60)
    print('【测试集最终评估】')
    print('=' * 60)
    print(f'R^2 : {r2_score(valid["rul_real"], valid["pred_post"]):.6f}')

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sub = test_df_eval.reset_index(drop=True)
    ax.plot(sub['rul_real'].values, label='真实 RUL', color='orange', linewidth=2.5)
    ax.plot(sub['pred_post'].values, label='预测', color='green', linewidth=2)
    ax.set_title(f'测试单元: {sub["unit"].iloc[0]}')
    ax.legend()
    plt.show()
    print('全部完成！')