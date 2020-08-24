import operator

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA

# ========= (1) CONFIGURE SYSTEM ========= #
N = 2000  # number of points in the scan
N_Corrupted = 100  # first points are removed from processing as they are corrupted
fs = 4e6  # sampling rate
FuSca = 6.8808e-6  # normalization coeeficient
c0 = 3e8  # speed of light
N = N - N_Corrupted  # removing corrupted points
NrChn = 32  # number of antennas
Win2D = np.tile(np.hanning(N), (NrChn - 1, 1))  # hanning window
ScaWin = np.sum(Win2D[1, :])  # normalization
# removing antenna 17 as redundant
WinAnt = np.hanning(NrChn - 1)
NFFT = 2 ** 14  # fft length
fStop = 77e9
fStrt = 76e9
TRampUp = 512e-6  # Ramp Up duration
kf = (fStop - fStrt) / TRampUp
vRange = [i for i in range(NFFT - 1)]
vRange = np.divide(vRange, NFFT / (fs * c0 / (2 * kf)))  # range bins
RMin = 1.0
RMax = 10.0
RMinIdx, Val = min(enumerate(np.abs(np.subtract(vRange, RMin))), key=operator.itemgetter(1))
RMaxIdx, Val = min(enumerate(np.abs(np.subtract(vRange, RMax))), key=operator.itemgetter(1))
vRangeExt = vRange[RMinIdx:RMaxIdx]  # we will only look into this range
# Window function for receive channels
NFFTAnt = 1024  # FFR length for azimuth
ScaWinAnt = np.sum(WinAnt)
WinAnt2D = np.tile(WinAnt, (np.size(vRangeExt), 1))
vAngDeg = [np.float(i) for i in range(int(-NFFTAnt / 2), int(NFFTAnt / 2))]
vAngDeg = np.multiply(np.arcsin(np.divide(vAngDeg, NFFTAnt / 2)), 180.0 / np.pi)
M = 12


def synthetic_signal(M):
    # ======= (1) INPUT DATA FROM SIMULATIONS ======= #
    # Signal source directions
    az = 180 * np.random.random_sample(M) - 90  # Azimuths
    example_range = 10 * np.random.random_sample(M)  # range

    # print('Azimuth Values', '\n', az, '\n', 'Ranges', '\n', example_range, '\n')
    el = np.zeros(np.shape(az))  # Simple example: assume elevations zero
    positionsY = [0.0, 1.948, 3.896, 5.844, 7.792, 9.74, 11.688, 13.636, 15.584, 17.532, 19.48, 21.428, 23.376, 25.324,
                  27.272, 29.22, 31.168, 33.116, 35.064, 37.012, 38.96, 40.908, 42.856, 44.804, 46.752, 48.7, 50.648,
                  52.596, 54.544, 56.492, 58.44
                  ]
    r = []
    for i in range(NrChn - 1):
        r.append([0., positionsY[i] / 1000., 0.])

    # ========= (1a) RECEIVED SIGNAL ========= #
    # Wavenumber vectors (in units of wavelength/2)
    X1 = np.cos(np.multiply(az, np.pi / 180.)) * np.cos(np.multiply(el, np.pi / 180.))
    X2 = np.sin(np.multiply(az, np.pi / 180.)) * np.cos(np.multiply(el, np.pi / 180.))
    X3 = np.sin(np.multiply(el, np.pi / 180.))
    k = np.multiply([X1, X2, X3], 2 * np.pi / (c0 / ((fStop + fStrt) / 2)))

    # Matrix of array response vectors
    rk = np.dot(r, k)
    A = np.exp(np.multiply(rk, -1j))
    # Additive noise
    sigma2 = 1.0  # Noise variance
    n = np.sqrt(sigma2) * (np.random.rand(NrChn - 1, N) + 1j * np.random.rand(NrChn - 1, N)) / np.sqrt(2)
    # Received signal
    tt = np.linspace(1, N, N)
    tau = 2 * example_range / c0
    beatFrequency = kf * tau
    beatFrequencyTT = np.outer(tt, beatFrequency)
    m = np.sin(np.multiply(beatFrequencyTT, 2 * np.pi / fs))

    DataV31 = (np.dot(A, np.transpose(m)) + n)
    return az, DataV31


def real_signal():
    # =========  READ Cal DATA from File========= %
    datafile = 'music/caldata32_1.csv'
    calData = np.genfromtxt(datafile, delimiter=',')
    CalData = calData[:, 1] + 1j * calData[:, 2]
    mCalData = np.tile(CalData, (N, 1))
    # ========= READ INPUT DATA from File========= %
    datafile = 'music/calipeda1.csv'
    DataV = np.genfromtxt(datafile, delimiter=',')
    # applying calibration
    DataV32 = DataV * mCalData
    DataV31 = np.concatenate((DataV32[:, 0:16], DataV32[:, 17:32]), axis=1)

    return np.transpose(DataV31)


az, d31 = synthetic_signal(M)  # synthetic signa
d31 = real_signal()  # real signa


def music():
    positionsY = [0.0, 1.948, 3.896, 5.844, 7.792, 9.74, 11.688, 13.636, 15.584, 17.532, 19.48, 21.428, 23.376, 25.324,
                  27.272, 29.22, 31.168, 33.116, 35.064, 37.012, 38.96, 40.908, 42.856, 44.804, 46.752, 48.7, 50.648,
                  52.596, 54.544, 56.492, 58.44
                  ]
    r = []
    for i in range(NrChn - 1):
        r.append([0., positionsY[i] / 1000., 0.])

    Rxx = d31 * np.matrix.getH(np.asmatrix(d31)) / N

    #     Eigendecompose

    D, E = LA.eig(Rxx)

    idx = D.argsort()[::-1]
    lmbd = D[idx]  # Vector of sorted eigenvalues
    E = E[:, idx]  # Sort eigenvectors accordingly
    En = E[:, M:len(E)]  # Noise eigenvectors (ASSUMPTION: M IS KNOWN)

    # MUSIC search directions
    AzSearch = np.arange(-90, 90, 0.1)  # Azimuth values to search
    ElSearch = [0]  # placeholder, we do not do elevation

    # ========= (4a) RECEIVED SIGNAL ========= #
    # Wavenumber vectors (in units of wavelength/2)
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

    # Get spherical coordinates
    P = np.unravel_index(Z.argmin(), Z.shape)
    # print(AzSearch[P])

    return AzSearch, Z


if __name__ == '__main__':
    # clean up the mess
    plt.close("all")

    AzSearch, Z = music()
    # Range throught FFT
    RP = np.fft.fft(d31 * Win2D, NFFT, 1)
    RPExt = RP[:, RMinIdx:RMaxIdx]
    # Digital Beam Forming
    JOpt_s = np.multiply(RPExt, np.transpose(WinAnt2D))
    JOpt_f = np.fft.fft(JOpt_s, NFFTAnt, 0) / ScaWinAnt
    JOpt = np.fft.fftshift(JOpt_f, 0)

    # Display time Series
    plt.figure(10)
    plt.plot(d31[0, :])
    plt.xlabel('T (us)')
    plt.ylabel('V (V)')

    # Display range profile
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
    # # normalize cost function
    JdB = 20. * np.log10(abs(JOpt))
    JMax = np.max(JdB)
    JNorm = JdB - JMax
    JNorm[JNorm < -60.] = -60.
    # # generate polar plot
    ax.pcolormesh(mU, mRange, np.transpose(JNorm))
    # ax.pcolor(mU, mRange, np.transpose(JNorm))

    ###### Compare MUSIC and DBF #####
    fig40 = plt.figure(40, figsize=(9, 9))
    Zmax = np.max(np.log10(Z))
    Zmin = np.min(np.log10(Z))
    plt.plot(AzSearch, (Zmax - np.log10(Z)) / (Zmax - Zmin))  # music plot

    # for synthetic image drawing azimuth of the targets
    # for xc in az:
    #    plt.axvline(x=-xc)

    JOpt_min = np.min(np.log10(np.sum(np.abs(JOpt), 1)))
    JOpt_max = np.max(np.log10(np.sum(np.abs(JOpt), 1)))
    plt.plot(vAngDeg, -(JOpt_min - np.log10(np.sum(np.abs(JOpt), 1))) / (JOpt_max - JOpt_min))  # DBF plot
    plt.ioff()
    plt.show()
