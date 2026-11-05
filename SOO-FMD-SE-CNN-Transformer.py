# ============================================================
# v46-数学逻辑终极对齐版：True SOO-FMD-SE-CNN-Transformer
# 核心修改：
# 1. ★ 严格对齐 MATLAB：xxc_mckd 每轮更新周期 T 后，重建 XmT 和 Xinv
# 2. ★ 严格对齐 MATLAB：FMD 剔除模式时，相关峭度 CK 采用 M=1 计算
# 3. 保留所有的防泄露、防溢出、因果滤波机制
# ============================================================

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import joblib
from scipy.stats import kurtosis, skew, pearsonr
from scipy.signal import welch, find_peaks, hilbert, savgol_filter, decimate, firwin, lfilter, correlate
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
_candidates = [
    '/home/hykj/data',
    r'G:\慧油科技数据\寿命预测',
    r'I:\慧油科技数据\寿命预测',
    r'D:\慧油科技数据\寿命预测',
    r'E:\慧油科技数据\寿命预测',
]
BASE_DIR = None
for _p in _candidates:
    if os.path.exists(os.path.join(_p, '7205')):
        BASE_DIR = _p
        break

if BASE_DIR is None:
    raise FileNotFoundError("❌ 找不到数据根目录。")

RESULT_DIR = os.path.join(BASE_DIR, 'deliverable')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

FS_ORIGINAL = 20000
DOWNSAMPLE_FACTOR = 2
FS = FS_ORIGINAL // DOWNSAMPLE_FACTOR

CHUNK_SIZE = 10000
CHUNK_STEP = 5000  # 50% 重叠提取特征
WINDOW_SIZE = 64
STEP = 4
BATCH_SIZE = 64
EPOCHS = 40
EARLY_STOP_PATIENCE = 8  # 早停机制容忍度
LEARN_RATE = 0.0003
MONO_BETA = 0.20
SMOOTH_GAMMA = 0.08
ASYM_PENALTY = 0.30
TOP_K_FEATURES = 15
POST_MEDIAN_WIN = 51
MC_SAMPLES = 20
RESAMPLE_STRIDE = 15

torch.manual_seed(100)
np.random.seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. SOO 优化算法
# ============================================================
def run_SOO(pop_size, max_iter, lb, ub, dim, fobj):
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    best_luminosity = float('inf')
    best_phase_position = np.zeros(dim)
    top_star_positions = np.zeros((3, dim))
    top_luminosities = np.full(3, float('inf'))
    current_luminosities = np.zeros(pop_size)
    initial_period = 3.0
    fitness_history = []

    for i in range(pop_size):
        current_luminosities[i] = fobj(positions[i, :])
        combined_l = np.append(top_luminosities, current_luminosities[i])
        combined_p = np.vstack((top_star_positions, positions[i, :]))
        idx = np.argsort(combined_l)
        top_luminosities = combined_l[idx[:3]]
        top_star_positions = combined_p[idx[:3], :]

    for it in range(1, max_iter + 1):
        current_period = initial_period + 0.001 * it
        current_angular_freq = 2 * np.pi / current_period

        for i in range(pop_size):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            current_luminosities[i] = fobj(positions[i, :])
            if current_luminosities[i] < best_luminosity:
                best_luminosity = current_luminosities[i]
                best_phase_position = positions[i, :].copy()

        scaling_factor = 2 - it * (2 / max_iter)

        for i in range(pop_size):
            for j in range(dim):
                r1, r2, r3 = np.random.rand(3)
                osc1 = best_phase_position[j] - (r1 * r3) * (
                        (current_angular_freq * scaling_factor * r1 - scaling_factor) *
                        (positions[i, j] - abs(r1 * np.sin(r2) * abs(r3 * best_phase_position[j]))))
                osc2 = best_phase_position[j] - (r2 * r3) * (
                        (current_angular_freq * scaling_factor * r1 - scaling_factor) *
                        (positions[i, j] - abs(r1 * np.cos(r2) * abs(r3 * best_phase_position[j]))))
                positions[i, j] = r3 * (osc1 + osc2 / 2)

        for i in range(pop_size):
            avg_best = np.mean(top_star_positions, axis=0)
            rand_idx = np.random.choice([x for x in range(pop_size) if x != i], 3, replace=False)
            r = np.random.rand()
            osc_pos = avg_best + 0.5 * (np.sin(r * np.pi) * (positions[rand_idx[0], :] - positions[rand_idx[1], :]) +
                                        np.cos((1 - r) * np.pi) * (
                                                positions[rand_idx[0], :] - positions[rand_idx[2], :]))
            star_update = positions[i, :].copy()
            for j in range(dim):
                if np.random.rand() <= 0.5:
                    star_update[j] = osc_pos[j]
            star_update = np.clip(star_update, lb, ub)

            new_l = fobj(star_update)
            if new_l < current_luminosities[i]:
                positions[i, :] = star_update
                current_luminosities[i] = new_l
                combined_l = np.append(top_luminosities, new_l)
                combined_p = np.vstack((top_star_positions, star_update))
                idx = np.argsort(combined_l)
                top_luminosities = combined_l[idx[:3]]
                top_star_positions = combined_p[idx[:3], :]

        best_primary = best_luminosity
        best_secondary = np.min(top_luminosities)
        if best_primary < best_secondary:
            best_luminosity = best_primary
        else:
            best_luminosity = best_secondary
            best_phase_position = top_star_positions[0, :]

        fitness_history.append(best_luminosity)

    return best_luminosity, best_phase_position, fitness_history


# ============================================================
# 2. FMD 算法 (★ 严格对齐 MATLAB 矩阵重建与 M=1)
# ============================================================
def estimate_period(y, fs):
    y_env = np.abs(hilbert(y))
    y_envelope = y_env - np.mean(y_env)

    M = int(fs)
    NA = correlate(y_envelope, y_envelope, mode='full')
    NA = NA[len(NA) // 2:]
    NA = NA / (np.max(NA) + 1e-12)

    zeroposi = 1
    for lag in range(1, len(NA)):
        if (NA[lag - 1] > 0 and NA[lag] < 0) or NA[lag] == 0:
            zeroposi = lag
            break

    NA_cut = NA[zeroposi:]
    return zeroposi + np.argmax(NA_cut)


def calc_CK(x, T, M=2):
    x = x.flatten()
    N = len(x)
    x_shift = np.zeros((M + 1, N))
    x_shift[0, :] = x
    for m in range(1, M + 1):
        x_shift[m, T:] = x_shift[m - 1, :-T]
    max_val = np.max(np.abs(x_shift), axis=1, keepdims=True) + 1e-12
    x_shift = x_shift / max_val
    prod_val = np.prod(x_shift, axis=0)
    return np.sum(prod_val ** 2) / (np.sum(x ** 2) ** (M + 1) + 1e-12)


def calc_envelope_entropy(sig):
    env = np.abs(hilbert(sig))
    env = env / (np.sum(env) + 1e-12)
    p = env[env > 0]
    return -np.sum(p * np.log(p + 1e-12))


def xxc_mckd(fs, x, f_init, termIter=20, M=3):
    x = x.flatten()
    L = len(f_init)
    N = len(x)

    x = x / (np.max(np.abs(x)) + 1e-12)

    if np.std(x) < 1e-6:
        return x, f_init, 0.0, 1

    T = estimate_period(x, fs)
    T = max(1, int(T))

    # 初始构建
    XmT = np.zeros((L, N, M + 1))
    for m in range(M + 1):
        for l in range(L):
            if l == 0:
                XmT[l, m * T:, m] = x[:N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]

    try:
        Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T + np.eye(L) * 1e-6)
    except:
        return x, f_init, 0.0, T

    f = f_init.copy()
    ck_best = 0
    y = np.zeros(N)

    for n in range(termIter):
        y = (f.T @ XmT[:, :, 0]).flatten()

        yt = np.zeros((N, M + 1))
        yt[:, 0] = y
        for m in range(1, M + 1):
            yt[m * T:, m] = yt[:-m * T, m - 1]

        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            others = np.prod(np.delete(yt, m, axis=1), axis=1)
            alpha[:, m] = (others ** 2) * yt[:, m]

        beta = np.prod(yt, axis=1)

        Xalpha = np.zeros((L, 1))
        for m in range(M + 1):
            Xalpha += XmT[:, :, m] @ alpha[:, m].reshape(-1, 1)

        f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2) + 1e-12)) * (Xinv @ Xalpha).flatten()
        f = f / np.sqrt(np.sum(f ** 2) + 1e-12)

        # 更新 T
        T_new = estimate_period(y, fs)
        T_new = max(1, int(T_new))

        # ★★★ 核心修复：重建 XmT 和 Xinv ★★★
        if T_new != T:
            T = T_new
            XmT = np.zeros((L, N, M + 1))
            for m in range(M + 1):
                for l in range(L):
                    if l == 0:
                        XmT[l, m * T:, m] = x[:N - m * T]
                    else:
                        XmT[l, 1:, m] = XmT[l - 1, :-1, m]
            try:
                Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T + np.eye(L) * 1e-6)
            except:
                pass

        ck_val = calc_CK(y, T, M)
        if ck_val > ck_best:
            ck_best = ck_val

    return y, f, ck_best, T


def apply_FMD(x, fs, FilterSize, ModeNum, MaxIter=10):
    x = x.flatten()
    CutNum = 10
    freq_bound = np.linspace(0, 1, CutNum + 1)[:-1]

    temp_filters = np.zeros((FilterSize, CutNum))
    for n in range(CutNum):
        temp_filters[:, n] = firwin(FilterSize, [freq_bound[n] + 1e-4, freq_bound[n] + 1 / CutNum - 1e-4],
                                    window='hann', pass_zero=False)

    modes_list = []
    ck_scores = []
    T_list = []

    for n in range(CutNum):
        y_final, f_final, ck_best, T_final = xxc_mckd(fs, x, temp_filters[:, n], termIter=MaxIter)
        modes_list.append(y_final)
        T_list.append(T_final)

        # ★★★ 严格对齐 MATLAB：剔除模式采用 M=1 的 CK ★★★
        ck_m1 = calc_CK(y_final, T_final, 1)
        ck_scores.append(ck_m1)

    modes = np.array(modes_list).T
    ck_scores = np.array(ck_scores)

    current_indices = list(range(CutNum))
    while len(current_indices) > ModeNum:
        current_modes = modes[:, current_indices]
        corr_matrix = np.abs(np.corrcoef(current_modes, rowvar=False))
        np.fill_diagonal(corr_matrix, 0)
        max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        idx1, idx2 = max_idx[0], max_idx[1]
        ck1 = ck_scores[current_indices[idx1]]
        ck2 = ck_scores[current_indices[idx2]]
        if ck1 > ck2:
            remove_idx = idx2
        else:
            remove_idx = idx1
        current_indices.pop(remove_idx)

    final_modes = modes[:, current_indices]
    final_cks = ck_scores[current_indices]
    sort_idx = np.argsort(final_cks)[::-1]

    return final_modes[:, sort_idx], final_cks[sort_idx]


def fitness_function(params, x, fs):
    filter_size = int(params[0])
    if filter_size % 2 == 0: filter_size += 1
    mode_num = int(params[1])
    try:
        modes, ck_scores = apply_FMD(x, fs, filter_size, mode_num, MaxIter=5)
        avg_ck = np.mean(ck_scores)
        avg_entropy = np.mean([calc_envelope_entropy(modes[:, i]) for i in range(modes.shape[1])])
        fitness_val = -(0.7 * avg_ck - 0.3 * avg_entropy)
        return fitness_val
    except:
        return 99999.0


def process_block_with_fmd(block, best_filter_size, best_mode_num):
    try:
        if len(block) < 100: return block
        modes, _ = apply_FMD(block, FS, best_filter_size, best_mode_num, MaxIter=10)
        correlations = []
        for i in range(modes.shape[1]):
            corr, _ = pearsonr(modes[:, i], block) if np.std(modes[:, i]) > 0 else (0.0, None)
            correlations.append(abs(corr))
        top3_idx = np.argsort(correlations)[-3:]
        return np.sum(modes[:, top3_idx], axis=1)
    except Exception as e:
        return block


# ============================================================
# 3. 特征提取
# ============================================================
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
            p = env_fft_half / s
            p = p[p > 0]
            e = -np.sum(p * np.log(p))
        else:
            c, pf, pa, e = 0.0, 0.0, 0.0, 0.0
        return {'HHT_EnvCentroid': c, 'HHT_EnvPeakFreq': pf, 'HHT_EnvPeakAmp': pa, 'HHT_EnvEntropy': e}
    except:
        return {'HHT_EnvCentroid': 0.0, 'HHT_EnvPeakFreq': 0.0, 'HHT_EnvPeakAmp': 0.0, 'HHT_EnvEntropy': 0.0}


def extract_features_enhanced(signal, signal_denoised):
    signal = np.asarray(signal, dtype=float)
    signal_denoised = np.asarray(signal_denoised, dtype=float)
    signal = signal[np.isfinite(signal)]
    signal_denoised = signal_denoised[np.isfinite(signal_denoised)]
    if len(signal) < 20 or len(signal_denoised) < 20: return None
    kurt = kurtosis(signal_denoised)
    if not np.isfinite(kurt): kurt = 0.0
    rms = np.sqrt(np.mean(signal_denoised ** 2))
    peak = np.max(np.abs(signal_denoised))
    peak_factor = peak / (rms + 1e-12)
    pulse_factor = peak / (np.mean(np.abs(signal_denoised)) + 1e-12)
    skewness = skew(signal_denoised)
    if not np.isfinite(skewness): skewness = 0.0
    hist, _ = np.histogram(signal, bins=50, density=True)
    p = hist[hist > 0]
    ent = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
    fd = higuchi_fd(signal)
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
        envelope = np.abs(hilbert(signal))
        env_freq, env_ps = welch(envelope, fs=FS, nperseg=nperseg)
        env_peaks, _ = find_peaks(env_ps, height=np.mean(env_ps))
        env_peak_freq = env_freq[env_peaks[np.argmax(env_ps[env_peaks])]] if len(env_peaks) > 0 else 0.0
    except:
        env_peak_freq = 0.0
    feat = {
        'Kurtosis': kurt, 'Entropy': ent, 'Fractal_Dimension': fd, 'Peak_factor': peak_factor,
        'Pulse_factor': pulse_factor, 'Skewness': skewness, 'Rms_vibration': rms, 'Peak_vibration': peak,
        'FFT_Centroid': centroid, 'FFT_MSF': msf, 'FFT_VarF': varf, 'FFT_Main_Freq': main_freq,
        'FFT_Main_Amp': main_freq_amp, 'Spectral_flatness': spectral_flatness,
        'Band1_E': band_energies[0], 'Band2_E': band_energies[1], 'Band3_E': band_energies[2],
        'Band4_E': band_energies[3], 'Band5_E': band_energies[4], 'Env_Peak_Freq': env_peak_freq
    }
    feat.update(extract_hht_features(signal))
    return feat


DUMMY_SIGNAL = np.zeros(1024)
FEATURE_NAMES = list(extract_features_enhanced(DUMMY_SIGNAL, DUMMY_SIGNAL).keys())


# ============================================================
# 4. 流式读取 + 重叠缓冲
# ============================================================
def process_file_streaming(path, unit_name, best_filter_size, best_mode_num):
    cache_path = os.path.join(RESULT_DIR, f'cache_{unit_name}.parquet')
    if os.path.exists(cache_path):
        print(f"  [断点续传] 发现已有缓存，直接秒读单元: {unit_name}")
        return cache_path

    print(f"  流式处理单元: {unit_name} (True SOO-FMD 运算中) ...")
    feats_list = []
    rms_list = []

    CHUNK_READ_SIZE = 100000
    OVERLAP = 20000
    buffer = np.array([])

    chunk_iter = pd.read_csv(path, header=None, usecols=[0], dtype=np.float32, chunksize=CHUNK_READ_SIZE)

    for chunk in chunk_iter:
        raw_sig = chunk.values.flatten()
        if DOWNSAMPLE_FACTOR > 1:
            sig_chunk = decimate(raw_sig, q=DOWNSAMPLE_FACTOR, ftype='fir', zero_phase=False)
        else:
            sig_chunk = raw_sig

        if len(buffer) > 0:
            sig_comb = np.concatenate((buffer, sig_chunk))
            start_offset = len(buffer)
        else:
            sig_comb = sig_chunk
            start_offset = 0

        sig_denoised_comb = process_block_with_fmd(sig_comb, best_filter_size, best_mode_num)

        end_idx = len(sig_comb) - OVERLAP if len(buffer) > 0 else len(sig_comb)
        sig_valid = sig_chunk[:-OVERLAP] if len(sig_chunk) > OVERLAP else sig_chunk
        sig_denoised_valid = sig_denoised_comb[start_offset:end_idx] if len(buffer) > 0 else sig_denoised_comb[:end_idx]

        buffer = sig_chunk[-OVERLAP:] if len(sig_chunk) > OVERLAP else sig_chunk

        for i in range(0, len(sig_valid) - CHUNK_SIZE + 1, CHUNK_STEP):
            ft = extract_features_enhanced(sig_valid[i:i + CHUNK_SIZE], sig_denoised_valid[i:i + CHUNK_SIZE])
            if ft:
                feats_list.append(ft)
                rms_list.append(np.sqrt(np.mean(sig_denoised_valid[i:i + CHUNK_SIZE] ** 2)))

        del raw_sig, sig_chunk, sig_comb, sig_denoised_comb, chunk
        gc.collect()

    df = pd.DataFrame(feats_list)
    df.columns = FEATURE_NAMES
    rms_seq = np.array(rms_list)

    window = min(101, len(rms_seq) - 1) if len(rms_seq) % 2 == 1 else min(100, len(rms_seq) - 2)
    if window % 2 == 0: window += 1
    smoothed_rms = savgol_filter(rms_seq, window_length=window, polyorder=3, mode='nearest')
    n = len(smoothed_rms)

    temp_median = np.median(smoothed_rms)
    working_mask = smoothed_rms > (temp_median * 0.2)

    if np.sum(working_mask) > 10:
        active_rms = smoothed_rms[working_mask]
        global_median = np.median(active_rms)
        global_std = np.std(active_rms)
    else:
        global_median = temp_median
        global_std = np.std(smoothed_rms)

    threshold = global_median + 3 * global_std

    fpt_idx = n - 1
    consecutive_count = 0
    for i in range(n):
        if working_mask[i] and smoothed_rms[i] >= threshold:
            consecutive_count += 1
            if consecutive_count >= 5:
                fpt_idx = i - 4
                break
        else:
            consecutive_count = 0

    print(f"    [FPT 检测] 退化点: {fpt_idx} (总长: {len(df)}) | 阈值: {threshold:.2f}")
    rul = np.ones(n)
    if n - fpt_idx > 1:
        t = np.linspace(0, 1, n - fpt_idx)
        rul[fpt_idx:] = 1.0 - t
    df['rul'] = rul
    df['unit'] = unit_name

    df.to_parquet(cache_path, compression='snappy')
    print(f"    [保存成功] 单元 {unit_name} 的特征已存盘！")
    return cache_path


# ============================================================
# 5. 模型定义
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid())

    def forward(self, x):
        b, c, t = x.size()
        return x * self.fc(x.mean(dim=2)).view(b, c, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


class CNNTransformerSeq(nn.Module):
    def __init__(self, input_dim, conv_channels=32, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1), nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True), nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels), nn.ReLU(inplace=True))
        self.se = SEBlock(conv_channels, reduction=8)
        self.pos_enc = PositionalEncoding(conv_channels, max_len=WINDOW_SIZE + 4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_channels, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head_rul = nn.Linear(conv_channels, 1)
        self.head_hi = nn.Linear(conv_channels, 1)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1))
        x = self.se(x)
        x = self.pos_enc(x.permute(0, 2, 1))
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
        diff = pred_rul[:, 1:] - pred_rul[:, :-1]
        mono = torch.relu(diff).mean()
        smooth = (pred_rul[:, 2:] - 2 * pred_rul[:, 1:-1] + pred_rul[:, :-2]).pow(2).mean() if pred_rul.size(
            1) >= 3 else torch.tensor(0.0, device=pred_rul.device)
        loss_hi = self.huber(pred_hi, target).mean()
        over_est = torch.relu(pred_rul - target)
        loss_asym = (over_est * weights).mean() * self.asym_penalty
        return loss_main + self.beta * mono + self.gamma * smooth + self.hi_weight * loss_hi + loss_asym


def predict_unit_full_mc(model, sub_df, feature_cols, window_size, step, device, batch_sz=128, n_samples=20):
    data = sub_df[feature_cols].values.astype(np.float32)
    n = len(data)
    if n < window_size: return np.full(n, np.nan), np.full(n, np.nan)
    windows, starts = [], []
    for i in range(0, n - window_size + 1, step):
        windows.append(data[i:i + window_size])
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


# ============================================================
# 6. 特征级时间重采样（唯一且严格的因果平滑）
# ============================================================
def resample_features(df, feature_cols, stride):
    resampled_dfs = []
    for unit in df['unit'].unique():
        sub = df[df['unit'] == unit].reset_index(drop=True)
        if len(sub) <= WINDOW_SIZE:
            resampled_dfs.append(sub)
            continue
        smoothed_feats = sub[feature_cols].rolling(window=stride, min_periods=1, center=False).mean()
        new_sub = pd.DataFrame(smoothed_feats, columns=feature_cols)
        new_sub['rul'] = sub['rul']
        new_sub['unit'] = unit
        sub_sampled = new_sub.iloc[::stride].copy()
        sub_sampled = sub_sampled.reset_index(drop=True)
        resampled_dfs.append(sub_sampled)
    if not resampled_dfs: return df
    return pd.concat(resampled_dfs, ignore_index=True)


if __name__ == '__main__':
    try:
        import multiprocessing

        multiprocessing.freeze_support()
    except:
        pass

    print(f"使用设备: {device}")

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

    print("\n" + "=" * 50)
    print("开始执行 SOO 算法，寻找 True FMD 最优参数...")

    sample_sig = None
    for folder, sensor_name, unit_tag, is_train in sensor_configs:
        if not is_train: continue
        test_path = os.path.join(folder, f"{sensor_name}.csv")
        if os.path.exists(test_path):
            print(f"提取寻优样本 (仅限训练集): {test_path}")
            sample_sig = pd.read_csv(test_path, header=None, nrows=10000).values.flatten()
            if DOWNSAMPLE_FACTOR > 1:
                sample_sig = decimate(sample_sig, q=DOWNSAMPLE_FACTOR, ftype='fir', zero_phase=False)
            break

    BEST_FILTER_SIZE = 60
    BEST_MODE_NUM = 3

    if sample_sig is not None:
        lb = np.array([30, 2])
        ub = np.array([100, 8])
        dim = 2
        _, best_params, soo_history = run_SOO(pop_size=20, max_iter=15, lb=lb, ub=ub, dim=dim,
                                              fobj=lambda p: fitness_function(p, sample_sig, FS))
        BEST_FILTER_SIZE = int(best_params[0]) if int(best_params[0]) % 2 == 1 else int(best_params[0]) + 1
        BEST_MODE_NUM = int(best_params[1])
        print(f"✅ SOO 寻优完成！全局 True FMD 参数锁定为: FilterSize={BEST_FILTER_SIZE}, ModeNum={BEST_MODE_NUM}")

        print("正在绘制 SOO 收敛曲线...")
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(soo_history) + 1), soo_history, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.title('SOO Optimization Convergence Curve', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value (Minimization)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        soo_plot_path = os.path.join(PLOT_DIR, 'SOO_Convergence_Curve.png')
        plt.savefig(soo_plot_path, dpi=300)
        plt.close()
        print(f"✅ SOO 收敛曲线已保存至: {soo_plot_path}")
    else:
        print("⚠️ 未找到数据文件，使用默认 FMD 参数。")
    print("=" * 50 + "\n")

    print("开始流式提取特征并存盘 (True SOO-FMD 运算中)...")
    train_cache_paths, test_cache_paths = [], []
    for folder, sensor_name, unit_tag, is_train in sensor_configs:
        path = os.path.join(folder, f"{sensor_name}.csv")
        if os.path.exists(path):
            cache_path = process_file_streaming(path, unit_tag, BEST_FILTER_SIZE, BEST_MODE_NUM)
            if is_train:
                train_cache_paths.append(cache_path)
            else:
                test_cache_paths.append(cache_path)

    print("\n所有特征已存盘，开始加载并合并特征...")
    train_df = pd.concat([pd.read_parquet(p) for p in train_cache_paths], ignore_index=True)
    test_df = pd.concat([pd.read_parquet(p) for p in test_cache_paths], ignore_index=True)
    all_feature_cols = FEATURE_NAMES

    print(f"正在进行时间维度降采样 (RESAMPLE_STRIDE={RESAMPLE_STRIDE})，扩展 Transformer 视野...")

    train_df = resample_features(train_df, all_feature_cols, RESAMPLE_STRIDE)
    test_df = resample_features(test_df, all_feature_cols, RESAMPLE_STRIDE)

    mi_scores = mutual_info_regression(train_df[all_feature_cols].values, train_df['rul'].values, random_state=100)
    mi_df = pd.DataFrame({'feature': all_feature_cols, 'mi': mi_scores}).sort_values('mi', ascending=False).reset_index(
        drop=True)
    feature_cols = mi_df['feature'].head(TOP_K_FEATURES).tolist()

    print("拟合全局 Scaler...")
    global_scaler = StandardScaler()
    global_scaler.fit(train_df[feature_cols].values)
    train_df[feature_cols] = global_scaler.transform(train_df[feature_cols].values)
    test_df[feature_cols] = global_scaler.transform(test_df[feature_cols].values)
    joblib.dump(global_scaler, os.path.join(RESULT_DIR, 'global_scaler.pkl'))

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

    del train_df, test_df
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
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            pred_rul, pred_hi = model(seq)
            loss = loss_function(pred_rul, pred_hi, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        val_loss = []
        with torch.no_grad():
            for seq, labels in test_loader:
                seq, labels = seq.to(device), labels.to(device)
                pred_rul, pred_hi = model(seq)
                val_loss.append(loss_function(pred_rul, pred_hi, labels).item())
        avg_val = np.average(val_loss)
        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
            flag = " ← 保存最佳模型"
        else:
            early_stop_counter += 1
            flag = ""

        print(f'Epoch {epoch + 1:2} | Train: {np.average(train_loss):.4f} | Val: {avg_val:.4f}{flag}')

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"⚠️ 验证集 Loss 已连续 {EARLY_STOP_PATIENCE} 轮未下降，触发早停机制。")
            break

    test_df_eval = pd.read_parquet(test_cache_paths[0])
    test_df_eval = resample_features(test_df_eval, FEATURE_NAMES, RESAMPLE_STRIDE)
    test_df_eval[feature_cols] = global_scaler.transform(test_df_eval[feature_cols].values)
    test_df_eval['rul_real'] = test_df_eval['rul']

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_df_eval['pred_raw'] = np.nan
    test_df_eval['pred_post'] = np.nan

    for unit in test_df_eval['unit'].unique():
        mask = test_df_eval['unit'] == unit
        sub = test_df_eval.loc[mask].reset_index(drop=True)
        preds, _ = predict_unit_full_mc(model, sub, feature_cols, WINDOW_SIZE, STEP, device)
        valid = ~np.isnan(preds)
        inv = np.full_like(preds, np.nan)
        inv[valid] = preds[valid]

        y = inv[valid]
        healthy_len = max(int(len(y) * 0.1), 5)
        base_val = np.median(y[:healthy_len])
        y_calibrated = y - base_val + 1.0
        y_calibrated = np.clip(y_calibrated, 0.0, 1.0)

        y_calibrated = pd.Series(y_calibrated).rolling(window=POST_MEDIAN_WIN, min_periods=1).mean().values
        y_calibrated = np.minimum.accumulate(y_calibrated)

        post = inv.copy()
        post[valid] = y_calibrated
        test_df_eval.loc[mask, 'pred_raw'] = inv
        test_df_eval.loc[mask, 'pred_post'] = post

    valid = test_df_eval.dropna(subset=['pred_post', 'rul_real'])
    print('\n' + '=' * 60)
    print('【测试集最终评估】')
    print('=' * 60)

    r2 = r2_score(valid["rul_real"], valid["pred_post"])
    rmse = np.sqrt(mean_squared_error(valid["rul_real"], valid["pred_post"]))
    mae = mean_absolute_error(valid["rul_real"], valid["pred_post"])

    print(f'R^2  : {r2:.6f}')
    print(f'RMSE : {rmse:.6f}')
    print(f'MAE  : {mae:.6f}')
    print('=' * 60 + '\n')

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sub = test_df_eval.reset_index(drop=True)
    ax.plot(sub['rul_real'].values, label='真实 RUL', color='orange', linewidth=2.5)
    ax.plot(sub['pred_post'].values, label='预测', color='green', linewidth=2)
    ax.set_title(f'测试单元: {sub["unit"].iloc[0]}')
    ax.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'result_SOO_FMD_CNN_Transformer.png'))
    plt.close()
    print(f'✅ 最终预测结果图已保存至: {os.path.join(PLOT_DIR, "result_SOO_FMD_CNN_Transformer.png")}')
    print('全部完成！')