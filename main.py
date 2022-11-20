#  Copyright [2022] [庄子奇]
#  Licensed under the WTFPL
#  Everyone is permitted to copy and distribute verbatim or modified
#  copies of this license document, and changing it is allowed as long
#  as the name is changed.

import numpy as np
from scipy.signal import iirdesign, freqz, remez
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Fs
    sample_frequency = 80000
    # fp
    passband_frequency = 15000
    # fs
    stopband_frequency = 20000
    # δ1
    delta_1 = 0.02
    # δ2
    delta_2 = 0.01
    # αp
    passband_loss = -20 * np.log10((1 - delta_1) / (1 + delta_1))
    # αs
    stopband_loss = 40
    # ωp
    discrete_passband_frequency = 2 * passband_frequency / sample_frequency
    # ωs
    discrete_stopband_frequency = 2 * stopband_frequency / sample_frequency
    # 过渡带宽度
    max_transition_band = (discrete_passband_frequency - discrete_stopband_frequency) * np.pi
    # 滤波器长度N matlab中(remezord)算出来的不对，这是试出来的
    remez_numtaps = 34
    # 椭圆滤波器
    filter_type = 'ellip'

    # 计算椭圆滤波器iir
    system = iirdesign(discrete_passband_frequency,
                       discrete_stopband_frequency,
                       passband_loss,
                       stopband_loss)
    w, h = freqz(*system)
    (b, a) = system
    order = len(a) - 1
    print(f'椭圆滤波器阶数为：{order}')

    plt.figure(layout='constrained')
    plt.subplot(221)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('ellipse filter \n frequency response')
    plt.ylabel('Amplitude[db]')
    plt.xlabel('frequency[rad]')
    plt.grid()

    plt.subplot(222)
    plt.plot(w, np.angle(h))
    plt.title('ellipse filter \n phase frequency')
    plt.ylabel('phase[π]')
    plt.xlabel('frequency[rad]')
    plt.grid()

    # 计算remez法（等波纹逼近）fir滤波器
    taps = remez(remez_numtaps,
                 [0, passband_frequency, stopband_frequency, 0.5 * sample_frequency],
                 [1, 0],
                 fs=sample_frequency
                 )

    w, h = freqz(taps, [1])

    plt.subplot(223)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('remez filter \n frequency response')
    plt.ylabel('Amplitude[db]')
    plt.xlabel('frequency[rad]')
    plt.grid()

    plt.subplot(224)
    plt.plot(w, np.angle(h))
    plt.title('remez filter \n phase frequency')
    plt.ylabel('phase[π]')
    plt.xlabel('frequency[rad]')
    plt.grid()

    plt.show()
