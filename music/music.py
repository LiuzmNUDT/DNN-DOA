import operator

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA

# ========= (1) 参数设置部分 ========= #
N = 2000  # 扫描的点数
N_Corrupted = 100  # 删除过程中的第一个点，因为它们已经被破坏
fs = 4e6  # 采样率
FuSca = 6.8808e-6  # 归一化系数
c0 = 3e8  # 光速
N = N - N_Corrupted  # 移除已经损坏的点
NrChn = 32  # 天线数目
Win2D = np.tile(np.hanning(N), (NrChn - 1, 1))  # hanning 窗口 https://en.wikipedia.org/wiki/Hann_function
ScaWin = np.sum(Win2D[1, :])  # 归一化

WinAnt = np.hanning(NrChn - 1)
NFFT = 2 ** 14  # FFT的长度
fStop = 77e9
fStrt = 76e9
TRampUp = 512e-6  # 增加持续时间
kf = (fStop - fStrt) / TRampUp
vRange = [i for i in range(NFFT - 1)]
vRange = np.divide(vRange, NFFT / (fs * c0 / (2 * kf)))  # 距离的范围
RMin = 1.0
RMax = 10.0
RMinIdx, Val = min(enumerate(np.abs(np.subtract(vRange, RMin))), key=operator.itemgetter(1))
RMaxIdx, Val = min(enumerate(np.abs(np.subtract(vRange, RMax))), key=operator.itemgetter(1))
vRangeExt = vRange[RMinIdx:RMaxIdx]  # 只探究这个范围

# 接收通道的窗口函数
NFFTAnt = 1024  # 方位角的FFR长度
ScaWinAnt = np.sum(WinAnt)
WinAnt2D = np.tile(WinAnt, (np.size(vRangeExt), 1))
vAngDeg = [np.float(i) for i in range(int(-NFFTAnt / 2), int(NFFTAnt / 2))]
vAngDeg = np.multiply(np.arcsin(np.divide(vAngDeg, NFFTAnt / 2)), 180.0 / np.pi)
M = 12


def synthetic_signal(M):
    # ======= (1) 模拟输入数据 ======= #
    # 信号源方向
    az = 180 * np.random.random_sample(M) - 90  # 方位
    example_range = 10 * np.random.random_sample(M)  # 范围
    el = np.zeros(np.shape(az))  # 海拔
    # print('方位值：', az, '范围：', example_range, '海拔：', el)

    positionsY = [0.0, 1.948, 3.896, 5.844, 7.792, 9.74, 11.688, 13.636, 15.584, 17.532, 19.48, 21.428, 23.376, 25.324,
                  27.272, 29.22, 31.168, 33.116, 35.064, 37.012, 38.96, 40.908, 42.856, 44.804, 46.752, 48.7, 50.648,
                  52.596, 54.544, 56.492, 58.44]
    r = []
    for i in range(NrChn - 1):
        r.append([0., positionsY[i] / 1000., 0.])

    # ========= (1a) 接收信号 ========= #
    # 波数向量 (以波长的一般为单位)
    X1 = np.cos(np.multiply(az, np.pi / 180.)) * np.cos(np.multiply(el, np.pi / 180.))
    X2 = np.sin(np.multiply(az, np.pi / 180.)) * np.cos(np.multiply(el, np.pi / 180.))
    X3 = np.sin(np.multiply(el, np.pi / 180.))
    k = np.multiply([X1, X2, X3], 2 * np.pi / (c0 / ((fStop + fStrt) / 2)))

    # 数组响应向量矩阵
    rk = np.dot(r, k)
    A = np.exp(np.multiply(rk, -1j))

    # 加性噪声
    sigma2 = 1.0  # 噪声方差
    n = np.sqrt(sigma2) * (np.random.rand(NrChn - 1, N) + 1j * np.random.rand(NrChn - 1, N)) / np.sqrt(2)

    # 接收到的信号
    tt = np.linspace(1, N, N)
    tau = 2 * example_range / c0
    beatFrequency = kf * tau
    beatFrequencyTT = np.outer(tt, beatFrequency)
    m = np.sin(np.multiply(beatFrequencyTT, 2 * np.pi / fs))

    DataV31 = (np.dot(A, np.transpose(m)) + n)
    return az, DataV31


def real_signal():
    # =========  读取计算数据 ========= %
    datafile = 'music/caldata32_1.csv'
    calData = np.genfromtxt(datafile, delimiter=',')
    CalData = calData[:, 1] + 1j * calData[:, 2]
    mCalData = np.tile(CalData, (N, 1))

    # ========= 读取输入数据 ========= %
    datafile = 'music/calipeda1.csv'
    DataV = np.genfromtxt(datafile, delimiter=',')

    # 应用到校准
    DataV32 = DataV * mCalData
    DataV31 = np.concatenate((DataV32[:, 0:16], DataV32[:, 17:32]), axis=1)

    return np.transpose(DataV31)


az, d31 = synthetic_signal(M)  # 合成的标记
d31 = real_signal()  # real signa


def music():
    positionsY = [0.0, 1.948, 3.896, 5.844, 7.792, 9.74, 11.688, 13.636, 15.584, 17.532, 19.48, 21.428, 23.376, 25.324,
                  27.272, 29.22, 31.168, 33.116, 35.064, 37.012, 38.96, 40.908, 42.856, 44.804, 46.752, 48.7, 50.648,
                  52.596, 54.544, 56.492, 58.44]
    r = []
    for i in range(NrChn - 1):
        r.append([0., positionsY[i] / 1000., 0.])

    Rxx = d31 * np.matrix.getH(np.asmatrix(d31)) / N

    # 特征分解

    D, E = LA.eig(Rxx)

    idx = D.argsort()[::-1]
    lmbd = D[idx]  # 排好序的特征值向量
    E = E[:, idx]  # 相应地对特征向量进行排序
    En = E[:, M:len(E)]  # 假设M是未知情况下的噪声特征向量

    # MUSIC 算法搜索方向
    AzSearch = np.arange(-90, 90, 0.1)  # 搜索方向的角度
    ElSearch = [0]  # 仅占位符

    # ========= (4a) 接收信号 ========= #
    # 波数向量（以波长的一般为单位）
    X1 = np.cos(np.multiply(AzSearch, np.pi / 180.))
    X2 = np.sin(np.multiply(AzSearch, np.pi / 180.))
    X3 = np.sin(np.multiply(AzSearch, 0.))
    kSearch = np.multiply([X1, X2, X3], 2 * np.pi / (c0 / ((fStop + fStrt) / 2)))
    ku = np.dot(r, kSearch)
    ASearch = np.exp(np.multiply(ku, -1j))
    chemodan = np.dot(np.transpose(ASearch), En)
    aac = np.absolute(chemodan)
    aad = np.square(aac)
    aae = np.sum(aad, 1)
    Z = aae

    # 获取球坐标
    P = np.unravel_index(Z.argmin(), Z.shape)
    # print(AzSearch[P])

    return AzSearch, Z


# 测试代码，仅在测试使起效
if __name__ == '__main__':
    plt.close("all")

    AzSearch, Z = music()

    # FFT范围
    RP = np.fft.fft(d31 * Win2D, NFFT, 1)
    RPExt = RP[:, RMinIdx:RMaxIdx]

    # 数字波束形成
    JOpt_s = np.multiply(RPExt, np.transpose(WinAnt2D))
    JOpt_f = np.fft.fft(JOpt_s, NFFTAnt, 0) / ScaWinAnt
    JOpt = np.fft.fftshift(JOpt_f, 0)

    # 展示时间序列
    plt.figure(10)
    plt.plot(d31[0, :])
    plt.xlabel('T (us)')
    plt.ylabel('V (V)')

    # 展示范围
    plt.figure(20)
    plt.plot(vRangeExt, 20. * np.log10(abs(RPExt[0, :])))
    plt.grid()
    plt.xlabel('R (m)')
    plt.ylabel('X (dBV)')

    # Range - Azimuth Polar Plot
    fig30 = plt.figure(30, figsize=(9, 9))

    # Positions for polar plot of cost function
    vU = vAngDeg * np.pi / 180.
    mU, mRange = np.meshgrid(vU, vRangeExt)
    ax = fig30.add_subplot(111, projection='polar')

    # normalize cost function
    JdB = 20. * np.log10(abs(JOpt))
    JMax = np.max(JdB)
    JNorm = JdB - JMax
    JNorm[JNorm < -60.] = -60.

    # 产生极坐标图
    ax.pcolormesh(mU, mRange, np.transpose(JNorm))
    # ax.pcolor(mU, mRange, np.transpose(JNorm))

    # 将MUSIC和DBF进行比较
    fig40 = plt.figure(40, figsize=(9, 9))
    Zmax = np.max(np.log10(Z))
    Zmin = np.min(np.log10(Z))
    plt.plot(AzSearch, (Zmax - np.log10(Z)) / (Zmax - Zmin))  # music plot

    JOpt_min = np.min(np.log10(np.sum(np.abs(JOpt), 1)))
    JOpt_max = np.max(np.log10(np.sum(np.abs(JOpt), 1)))
    plt.plot(vAngDeg, -(JOpt_min - np.log10(np.sum(np.abs(JOpt), 1))) / (JOpt_max - JOpt_min))  # DBF plot
    plt.ioff()
    plt.show()
