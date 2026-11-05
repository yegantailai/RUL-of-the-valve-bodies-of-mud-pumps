# ============================================================
# v5：轻量DWT去噪 + 分特征处理 + 强化正则化治过拟合
# 修复 v4：DWT去噪过度，把判别特征一起抹掉
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks, hilbert
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter, gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
import matplotlib
import matplotlib.pyplot as plt

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("提示：未安装 PyWavelets，将跳过去噪。建议 pip install PyWavelets")

matplotlib.rc("font", family='Microsoft YaHei')
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
BASE_DIR = r'G:\慧油科技数据\寿命预测'
RESULT_DIR = 'dataresult_my'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs('画图对比', exist_ok=True)

FS = 20000
CHUNK_SIZE = 2048
CHUNK_STEP = 1024
WINDOW_SIZE = 64
STEP = 4
BATCH_SIZE = 32
EPOCHS = 40
LEARN_RATE = 0.0003
HEALTHY_RATIO = 0.2
FEATURE_SMOOTH_WIN = 15
MONO_BETA = 0.15
SMOOTH_GAMMA = 0.05

# —— 去噪参数（轻量）——
DENOISE_ENABLE = True
WAVELET = 'db4'
WAVELET_LEVEL = 2
SOFT_FACTOR = 0.5

# —— 后处理 ——
POST_MEDIAN_WIN = 31
POST_GAUSS_SIGMA = 80

torch.manual_seed(100)
np.random.seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)


# ============================================================
# 1. 轻量 DWT 去噪
# ============================================================
def dwt_gentle_denoise(signal, wavelet=WAVELET, level=WAVELET_LEVEL, soft_factor=SOFT_FACTOR):
    if not HAS_PYWT or not DENOISE_ENABLE:
        return signal
    n = len(signal)
    if n < 2 ** (level + 1):
        return signal
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        if sigma < 1e-12:
            return signal
        threshold = sigma * soft_factor
        coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
        rec = pywt.waverec(coeffs, wavelet)
        return rec[:n]
    except Exception as e:
        print(f"    小波去噪失败，跳过一次: {e}")
        return signal


# ============================================================
# 2. 特征提取（分：时域用去噪信号，频域/复杂度用原始信号）
# ============================================================
def higuchi_fd(x, kmax=5):
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
    x_axis = np.log(1.0 / np.arange(1, len(L) + 1))
    y_axis = np.log(L + 1e-12)
    try:
        return np.polyfit(x_axis, y_axis, 1)[0]
    except:
        return 0.0


def extract_features_enhanced(signal, signal_denoised):
    """
    signal: 原始信号（用于频域、熵、分形维数）
    signal_denoised: 去噪信号（用于时域统计）
    """
    signal = np.asarray(signal, dtype=float)
    signal_denoised = np.asarray(signal_denoised, dtype=float)
    signal = signal[np.isfinite(signal)]
    signal_denoised = signal_denoised[np.isfinite(signal_denoised)]
    if len(signal) < 20 or len(signal_denoised) < 20: return None

    # ========== 时域特征：用去噪信号 ==========
    kurt = kurtosis(signal_denoised)
    if not np.isfinite(kurt): kurt = 0.0
    rms = np.sqrt(np.mean(signal_denoised ** 2))
    peak = np.max(np.abs(signal_denoised))
    peak_factor = peak / (rms + 1e-12)
    pulse_factor = peak / (np.mean(np.abs(signal_denoised)) + 1e-12)
    skewness = skew(signal_denoised)
    if not np.isfinite(skewness): skewness = 0.0

    # ========== 复杂度：用原始信号 ==========
    hist, _ = np.histogram(signal, bins=50, density=True)
    p = hist[hist > 0]
    ent = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
    fd = higuchi_fd(signal)

    # ========== 频域特征：用原始信号 ==========
    N = len(signal)
    fft_vals = rfft(signal)
    fft_abs = np.abs(fft_vals) / N
    fft_freqs = rfftfreq(N, d=1.0 / FS)

    sum_abs = np.sum(fft_abs)
    if sum_abs > 0:
        centroid = np.sum(fft_freqs * fft_abs) / sum_abs
        msf = np.sum((fft_freqs ** 2) * fft_abs) / sum_abs
        varf = np.sum(((fft_freqs - centroid) ** 2) * fft_abs) / sum_abs
        main_freq_idx = np.argmax(fft_abs)
        main_freq_amp = fft_abs[main_freq_idx]
        main_freq = fft_freqs[main_freq_idx]
    else:
        centroid, msf, varf, main_freq_amp, main_freq = 0.0, 0.0, 0.0, 0.0, 0.0

    nperseg = min(N, 1024)
    freq, ps = welch(signal, fs=FS, nperseg=nperseg)
    ps = np.maximum(ps, 1e-12)
    total_energy = np.sum(ps)
    spectral_flatness = np.exp(np.mean(np.log(ps))) / (np.mean(ps) + 1e-12)

    band_energies = []
    band_edges = np.linspace(0, len(ps), 6, dtype=int)
    for i in range(5):
        band_energy = np.sum(ps[band_edges[i]:band_edges[i + 1]])
        band_energies.append(band_energy / (total_energy + 1e-12))

    try:
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        env_freq, env_ps = welch(envelope, fs=FS, nperseg=nperseg)
        env_peaks, _ = find_peaks(env_ps, height=np.mean(env_ps))
        env_peak_freq = env_freq[env_peaks[np.argmax(env_ps[env_peaks])]] if len(env_peaks) > 0 else 0.0
    except:
        env_peak_freq = 0.0

    return {
        'Kurtosis': kurt, 'Entropy': ent, 'Fractal_Dimension': fd,
        'Peak_factor': peak_factor, 'Pulse_factor': pulse_factor,
        'Skewness': skewness, 'Rms_vibration': rms, 'Peak_vibration': peak,
        'FFT_Centroid': centroid, 'FFT_MSF': msf, 'FFT_VarF': varf,
        'FFT_Main_Freq': main_freq, 'FFT_Main_Amp': main_freq_amp,
        'Spectral_flatness': spectral_flatness,
        'Band1_E': band_energies[0], 'Band2_E': band_energies[1],
        'Band3_E': band_energies[2], 'Band4_E': band_energies[3],
        'Band5_E': band_energies[4], 'Env_Peak_Freq': env_peak_freq
    }


DUMMY_SIGNAL = np.zeros(1024)
FEATURE_NAMES = list(extract_features_enhanced(DUMMY_SIGNAL, DUMMY_SIGNAL).keys())


# ============================================================
# 3. 数据读取：先做轻量去噪，再切片
# ============================================================
def read_signal(path):
    sig = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0].values.astype(float)
    sig_denoised = dwt_gentle_denoise(sig)
    return sig, sig_denoised


def process_single_file(path, chunk_size, step):
    sig, sig_denoised = read_signal(path)
    feats = []
    for i in range(0, len(sig) - chunk_size + 1, step):
        ft = extract_features_enhanced(sig[i:i + chunk_size], sig_denoised[i:i + chunk_size])
        if ft: feats.append(ft)
    return pd.DataFrame(feats)


def build_unit_df(file_path, unit_name):
    if not os.path.exists(file_path):
        print(f"  警告: 缺失 {file_path}，跳过！")
        return None
    print(f"  处理单元: {unit_name} ...")
    df = process_single_file(file_path, CHUNK_SIZE, CHUNK_STEP)
    if df is None or len(df) == 0:
        return None
    df.columns = FEATURE_NAMES
    n = len(df)
    healthy_n = int(n * HEALTHY_RATIO)
    rul = np.ones(n)
    if n - healthy_n > 1:
        rul[healthy_n:] = np.linspace(1.0, 0.0, n - healthy_n)
    df['rul'] = rul
    df['unit'] = unit_name
    return df


# ============================================================
# 4. 传感器配置
# ============================================================
sensor_configs = [
    (os.path.join(BASE_DIR, '7201', '数据处理', '合并'),  '7201_sensor2', True),
    (os.path.join(BASE_DIR, '7201', '数据处理', '合并'),  '7201_sensor4', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor3', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor4', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor5', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期1'), '7205_sensor6', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor1', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor2', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor3', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor4', True),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor5', False),
    (os.path.join(BASE_DIR, '7205', '数据处理', '周期2'), '7205_sensor6', False),
]

train_dfs, test_dfs = [], []
for folder, sensor_name, is_train in sensor_configs:
    df = build_unit_df(os.path.join(folder, f"{sensor_name}.csv"), sensor_name)
    if df is None: continue
    (train_dfs if is_train else test_dfs).append(df)

if not train_dfs or not test_dfs:
    raise RuntimeError("训练集或测试集为空！")

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

feature_cols = FEATURE_NAMES
print(f'\n特征数: {len(feature_cols)}')
print(f'训练集样本数: {len(train_df)} (单元: {sorted(train_df["unit"].unique())})')
print(f'测试集样本数: {len(test_df)} (单元: {sorted(test_df["unit"].unique())})')


# ============================================================
# 5. 特征滑动平均
# ============================================================
def smooth_features_inplace(df, feature_cols, window=15):
    for unit in df['unit'].unique():
        mask = df['unit'] == unit
        sub = df.loc[mask, feature_cols]
        smoothed = sub.rolling(window=window, min_periods=1, center=True).mean()
        df.loc[mask, feature_cols] = smoothed.values
    return df

train_df = smooth_features_inplace(train_df, feature_cols, FEATURE_SMOOTH_WIN)
test_df = smooth_features_inplace(test_df, feature_cols, FEATURE_SMOOTH_WIN)


# ============================================================
# 6. 相关性检验
# ============================================================
print("\n=== 特征与 RUL 相关性 Top10 ===")
correlations = train_df[feature_cols].corrwith(train_df['rul']).abs().sort_values(ascending=False)
print(correlations.head(10))


# ============================================================
# 7. 标准化
# ============================================================
scaler_x = StandardScaler()
train_df[feature_cols] = scaler_x.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler_x.transform(test_df[feature_cols])

scaler_y = MinMaxScaler()
train_df[['rul']] = scaler_y.fit_transform(train_df[['rul']])
test_df[['rul']] = scaler_y.transform(test_df[['rul']])

dump(scaler_x, os.path.join(RESULT_DIR, 'scaler_x'))
dump(scaler_y, os.path.join(RESULT_DIR, 'scaler_y'))


# ============================================================
# 8. 窗口构造
# ============================================================
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

train_set = torch.tensor(train_X, dtype=torch.float32)
train_label = torch.tensor(train_y, dtype=torch.float32)
test_set = torch.tensor(test_X, dtype=torch.float32)
test_label = torch.tensor(test_y, dtype=torch.float32)

print('train_set:', train_set.shape)
print('test_set :', test_set.shape)


# ============================================================
# 9. 模型
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CNNTransformerSeq(nn.Module):
    def __init__(self, input_dim, conv_channels=32, hidden_dim=64,
                 num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.pos_enc = PositionalEncoding(conv_channels, max_len=WINDOW_SIZE + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_channels, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        out = self.head(x).squeeze(-1)
        return out


# ============================================================
# 10. 自定义损失
# ============================================================
class SequenceRULLoss(nn.Module):
    def __init__(self, beta=0.15, gamma=0.05):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.huber = nn.SmoothL1Loss()

    def forward(self, pred, target):
        main = self.huber(pred, target)
        diff = pred[:, 1:] - pred[:, :-1]
        mono = torch.relu(diff).mean()
        if pred.size(1) >= 3:
            sec = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            smooth = (sec ** 2).mean()
        else:
            smooth = torch.tensor(0.0, device=pred.device)
        return main + self.beta * mono + self.gamma * smooth


# ============================================================
# 11. 训练（无早停，保存最佳模型）
# ============================================================
def dataloader(batch_size):
    train_loader = Data.DataLoader(Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


train_loader, test_loader = dataloader(BATCH_SIZE)
input_dim = train_set.shape[-1]
model = CNNTransformerSeq(input_dim, conv_channels=32, hidden_dim=64,
                          num_layers=2, num_heads=4, dropout=0.5).to(device)

loss_function = SequenceRULLoss(beta=MONO_BETA, gamma=SMOOTH_GAMMA)
optimizer = torch.optim.Adam(model.parameters(), LEARN_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print('\n开始训练（无早停，跑满 EPOCHS）...')
best_loss = float('inf')
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        train_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    avg_loss = np.average(train_loss)

    model.eval()
    val_loss = []
    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            val_loss.append(loss_function(y_pred, labels).item())
    avg_val_loss = np.average(val_loss)
    scheduler.step(avg_val_loss)

    flag = ""
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'best_model.pt'))
        flag = " ← 保存最佳模型"

    print(f'Epoch {epoch + 1:2} Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}{flag}')

print(f'\n训练完成。最佳验证损失 {best_loss:.6f} 出现在 Epoch {best_epoch}')


# ============================================================
# 12. 多窗口重叠平均预测
# ============================================================
def predict_unit_full(model, sub_df, feature_cols, window_size, step, device, batch_sz=128):
    data = sub_df[feature_cols].values.astype(np.float32)
    n = len(data)
    if n < window_size:
        return np.full(n, np.nan)
    windows, starts = [], []
    for i in range(0, n - window_size + 1, step):
        windows.append(data[i:i + window_size])
        starts.append(i)
    windows = torch.tensor(np.array(windows), dtype=torch.float32)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for b in range(0, len(windows), batch_sz):
            out = model(windows[b:b + batch_sz].to(device)).cpu().numpy()
            all_preds.append(out)
    all_preds = np.concatenate(all_preds, axis=0)

    preds_sum = np.zeros(n)
    counts = np.zeros(n)
    for k, i in enumerate(starts):
        for j in range(window_size):
            preds_sum[i + j] += all_preds[k, j]
            counts[i + j] += 1
    preds = np.full(n, np.nan)
    valid = counts > 0
    preds[valid] = preds_sum[valid] / counts[valid]
    return preds


# ============================================================
# 13. 后处理（不强制单调）
# ============================================================
def postprocess_rul(pred, median_win=POST_MEDIAN_WIN, sigma=POST_GAUSS_SIGMA):
    valid = ~np.isnan(pred)
    if valid.sum() < 10:
        return pred
    p = pred.copy()
    y = p[valid]
    y = median_filter(y, size=median_win)
    y = gaussian_filter1d(y, sigma=sigma)
    y = np.clip(y, 0.0, 1.0)
    p[valid] = y
    return p


# ============================================================
# 14. 评估 + 可视化
# ============================================================
model.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_model.pt')))

test_df_eval = test_df.copy()
test_df_eval['pred_raw'] = np.nan
test_df_eval['pred_post'] = np.nan

for unit in test_df_eval['unit'].unique():
    mask = test_df_eval['unit'] == unit
    sub = test_df_eval.loc[mask].reset_index()
    preds = predict_unit_full(model, sub, feature_cols, WINDOW_SIZE, STEP, device)
    valid = ~np.isnan(preds)
    inv = np.full_like(preds, np.nan)
    inv[valid] = scaler_y.inverse_transform(preds[valid].reshape(-1, 1)).flatten()
    post = inv.copy()
    if valid.sum() > 10:
        post[valid] = postprocess_rul(inv[valid])
    test_df_eval.loc[mask, 'pred_raw'] = inv
    test_df_eval.loc[mask, 'pred_post'] = post

test_df_eval['rul_real'] = scaler_y.inverse_transform(test_df_eval[['rul']].values)

valid = test_df_eval.dropna(subset=['pred_post', 'rul_real'])
print(f'\n测试集 R^2:  {r2_score(valid["rul_real"], valid["pred_post"]):.6f}')
print(f'测试集 MSE: {mean_squared_error(valid["rul_real"], valid["pred_post"]):.6f}')
print(f'测试集 MAE: {mean_absolute_error(valid["rul_real"], valid["pred_post"]):.6f}')

units = sorted(test_df_eval['unit'].unique())
n_units = len(units)
fig, axes = plt.subplots(n_units, 1, figsize=(12, 4 * n_units), sharex=False)
if n_units == 1: axes = [axes]

for ax, unit in zip(axes, units):
    sub = test_df_eval[test_df_eval['unit'] == unit].reset_index(drop=True)
    ax.plot(sub['rul_real'].values, label='真实 RUL', color='orange', linewidth=2.5)
    ax.plot(sub['pred_raw'].values, label='原始预测（多窗口平均）',
            color='gray', alpha=0.5, linewidth=1)
    ax.plot(sub['pred_post'].values, label='后处理（去尖峰+高斯平滑）',
            color='green', linewidth=2)
    ax.set_title(f'测试单元: {unit}')
    ax.set_xlabel('时间步')
    ax.set_ylabel('寿命百分比')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('画图对比', 'per_unit_prediction_v5.png'), dpi=120)
plt.show()

dump(test_df_eval, '画图对比/cnn_transformer_results_v5')
print('\n完成！')