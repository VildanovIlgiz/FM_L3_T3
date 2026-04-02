import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==========================================
# 1. Пути к файлам и папкам
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

input_wav_path = os.path.join(BASE_DIR, "MUHA.wav")
output_wav_path = os.path.join(BASE_DIR, "MUHA_filtered.wav")

# ==========================================
# 2. Настройки оформления графиков
# ==========================================
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "mathtext.fontset": "dejavuserif"
})

# ==========================================
# 3. Загрузка аудиофайла
# ==========================================
fs, y = wavfile.read(input_wav_path)

# Если сигнал стерео, переводим в моно
if y.ndim > 1:
    y = y.mean(axis=1)

# Приводим к float64 и нормализуем
if np.issubdtype(y.dtype, np.integer):
    y = y.astype(np.float64) / np.iinfo(y.dtype).max
else:
    y = y.astype(np.float64)

N = len(y)
t = np.arange(N) / fs

# ==========================================
# 4. Преобразование Фурье исходного сигнала
# ==========================================
Y = np.fft.rfft(y)
freqs = np.fft.rfftfreq(N, d=1 / fs)
Y_mag = np.abs(Y)

# ==========================================
# 5. Жёсткий полосовой фильтр в частотной области
#    Оставляем диапазон 300-3000 Гц
# ==========================================
f_low = 300
f_high = 3000

H = np.zeros_like(freqs)
H[(freqs >= f_low) & (freqs <= f_high)] = 1.0

Y_filtered = Y * H
y_filtered = np.fft.irfft(Y_filtered, n=N)

# ==========================================
# 6. Преобразование Фурье фильтрованного сигнала
# ==========================================
Yf = np.fft.rfft(y_filtered)
Yf_mag = np.abs(Yf)

# ==========================================
# 7. Сравнительный график сигналов во времени
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(t, y, linewidth=1.0, alpha=0.85, label=r"$f(t)$")
plt.plot(t, y_filtered, linewidth=1.0, alpha=0.85, label=r"$f_{filtered}(t)$")

plt.xlabel("t, с")
plt.ylabel("f(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "time_signals_compare.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 8. Фурье-образ исходного сигнала: 0...4000 Гц
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(freqs, Y_mag, color="blue", linewidth=1.0)

plt.xlim(0, 4000)
plt.xlabel("ν, Гц")
plt.ylabel(r"$|\hat{f}(ν)|$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "fft_original_0_4000.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 9. Фурье-образ исходного сигнала: 0...1250 Гц
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(freqs, Y_mag, color="blue", linewidth=1.0)

plt.xlim(0, 1250)
plt.xlabel("ν, Гц")
plt.ylabel(r"$|\hat{f}(ν)|$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "fft_original_0_1250.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 10. Фурье-образ фильтрованного сигнала: 0...4000 Гц
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(freqs, Yf_mag, linewidth=1.0)

plt.xlim(0, 4000)
plt.xlabel("ν, Гц")
plt.ylabel(r"$|\hat{f}(ν)|$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "fft_filtered_0_4000.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 12. Сохранение фильтрованного звука
# ==========================================
max_val = np.max(np.abs(y_filtered))
if max_val > 0:
    y_out = y_filtered / max_val
else:
    y_out = y_filtered

y_out_int16 = (y_out * 32767).astype(np.int16)
wavfile.write(output_wav_path, fs, y_out_int16)

# ==========================================
# 11. Сравнение Фурье-образов до и после фильтрации
# ==========================================
plt.figure(figsize=(12, 6))

plt.plot(freqs, Y_mag, linewidth=1.0, alpha=0.85, label=r"$\hat{f}(\nu)$")
plt.plot(freqs, Yf_mag, linewidth=1.0, alpha=0.85, label=r"$\hat{f}_{filtered}(\nu)$")

plt.xlim(0, 4000)
plt.xlabel("ν, Гц")
plt.ylabel(r"$|\hat{f}(ν)|$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "fft_compare_0_4000.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()