import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

pd.options.display.max_columns = None

df = pd.read_csv('HYYDATE1.csv', names=list(range(1, 25)))


def solar(lat, tilt, azimuth, albedo, A, B, C, lamda, n, month):
    if not n:
        start = 0
        end = 12
    else:
        start = n - 1
        end = n

    phi = np.radians(lat)
    beta = np.radians(tilt)
    gamma = np.radians(azimuth)

    es = []
    dec_d = []
    h = np.array(list(np.arange(1, 25)) * (month[end] - month[start]))

    for i in range(month[start], month[end]):
        es.append(E(i+1))
        dec_d.append(declination(i+1))

    e = np.repeat(np.array(es), 24)
    # omega = (15 * (h + (lamda - 135) / 15 + e / 60) - 180) * np.pi / 180
    # ts = h + (lamda - 135) / 15 + e / 60
    tm = 12-(lamda-135)/15-e/60
    # omega = (ts-12) * np.pi / 12
    omega = (h-tm)*np.pi/12
    delta = np.repeat(np.array(dec_d), 24)

    # Calculate angle of incidence on tilted surface for every hour of the year
    cos_theta = np.sin(delta) * np.sin(phi) * np.cos(beta) - np.sin(delta) * np.cos(phi) * np.sin(beta) * np.cos(
        gamma) + np.cos(delta) * np.cos(phi) * np.cos(beta) * np.cos(omega) + np.cos(delta) * np.sin(phi) * np.sin(
        beta) * np.cos(gamma) * np.cos(omega) + np.cos(delta) * np.sin(beta) * np.sin(gamma) * np.sin(omega)
    
    # Calculate zenith angle for every hour of the year
    cos_theta_z = np.cos(phi) * np.cos(delta) * np.cos(omega) + np.sin(phi) * np.sin(delta)

    # Ratio of beam radiation on tilted surface to beam radiation on horizontal surface
    Rb = cos_theta / cos_theta_z

    Gt = B * Rb + C * (1 + np.cos(beta)) / 2 + A * albedo * (1 - np.cos(beta)) / 2
    # Gt = Rb
    return Gt


class Count:
    def __init__(self):
        self.n = -1

    def count(self):
        self.n += 1
        return self.n


def declination(n):
    gama = 2 * np.pi * (n - 1) / 365
    delta = (0.006918 - 0.399912 * np.cos(gama) + 0.070257 * np.sin(gama) - 0.006758 * np.cos(
        2 * gama) + 0.000907 * np.sin(2 * gama) - 0.002697 * np.cos(3 * gama) + 0.00148 * np.sin(3 * gama))  # radians
    # delta = -np.arcsin(0.39779 * np.cos(np.radians(0.98565 * (n + 10) + 1.914 * np.sin(np.radians(0.98565 * (n - 2))))))
    return delta


def E(n):
    B = 2 * np.pi * (n - 81) / 364
    e = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)  # min
    return e


def choice_month(df, n):
    """
    0 代表全部月份
    1, 2, 3, , , 12 代表想要查看的月份
    """
    H = df.iloc[pd.Index([9 * i for i in range(365)])]  # 平射
    Hb = df.iloc[pd.Index([(9 * i + 1) for i in range(365)])]  # 直射
    Hd = df.iloc[pd.Index([(9 * i + 2) for i in range(365)])]  # 散乱
    St = df.iloc[pd.Index([(9 * i + 3) for i in range(365)])]  # sun time

    H.loc[:, 'n'] = list(range(365))
    Hb.loc[:, 'n'] = list(range(365))
    Hd.loc[:, 'n'] = list(range(365))
    St.loc[:, 'n'] = list(range(365))

    H.set_index('n', inplace=True)
    Hb.set_index('n', inplace=True)
    Hd.set_index('n', inplace=True)
    St.set_index('n', inplace=True)

    month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]).cumsum()
    H_sum = H.sum(axis=1)
    #     Hb_sum = Hb.sum(axis=1)
    #     Hd_sum = Hd.sum(axis=1)

    for i in range(len(month) - 1):
        tmp = H_sum[month[i]: month[i + 1]].sort_values()[:0].index
        with pd.option_context('mode.chained_assignment', None):
            H.loc[tmp, :] = 0
            Hb.loc[tmp, :] = 0
            Hd.loc[tmp, :] = 0
            St.loc[tmp, :] = 0

    if n:
        start = month[n - 1]
        end = month[n]
    else:
        start = 0
        end = 365

    H = H.iloc[pd.Index([i for i in range(start, end)])]  # 平射
    Hb = Hb.iloc[pd.Index([(i) for i in range(start, end)])]  # 直射
    Hd = Hd.iloc[pd.Index([(i) for i in range(start, end)])]  # 散乱
    St = St.iloc[pd.Index([(i) for i in range(start, end)])]  # sun time

    H = np.array(H).ravel()
    Hb = np.array(Hb).ravel()
    Hd = np.array(Hd).ravel()
    St = np.array(St).ravel()

    df = pd.DataFrame(np.array([H, Hb, Hd]).T, columns=['平射', '直射', '散乱'])

    df['平射'] = df['平射'] * 0.01 * 10 ** 6 / 3600
    df['直射'] = df['直射'] * 0.01 * 10 ** 6 / 3600
    df['散乱'] = df['散乱'] * 0.01 * 10 ** 6 / 3600

    A = df['平射']
    B = df['直射']
    C = df['散乱']

    # Korea Jeji
    # PHI = 33.489
    # lamda = 126.498


    # PHI=31.5
    # lamda=130.5


    # Naha
    PHI = 26.207  # latitude
    lamda = 127.685  # longitude

    lat = PHI
    albedo = 0.2

    azimuth = 0
    azimuth_end = 360
    azimuth_range = azimuth_end - azimuth

    tilt = 0
    tilt_end = 90

    count = Count()
    for i in range(azimuth, azimuth_end + 1):
        Solar_radiation_year = []
        for tilt in range(tilt_end + 1):
            df[str(count.count())] = solar(lat, tilt, i, albedo, A, B, C, lamda, n, month)
        print(i, end=' ')
    print()

    test = df.iloc[:, 3:].agg(['sum'])
    z = np.array(test).reshape((361, 91)) / 100000

    X = np.arange(0, 361, 1)  # Azimuth
    Y = np.arange(0, tilt_end + 1, 1)  # Tilt_angle
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 10))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, z[X, Y], cmap=mpl.cm.coolwarm)

    ax.set_xlabel(r"Azimuth $A_a$ [°]", fontsize=18)
    ax.set_ylabel(r"Tilt angle $\beta$ [°]", fontsize=18)
    ax.set_zlabel(r"Generated energy $P_{pv}$ [MWh]", fontsize=18)

    plt.title("Solar", fontsize=18)
    print(f'''Ppv_max: {z[np.where(z==z.max())[0],np.where(z==z.max())[1]] / 1000000},
        Azimuth: {np.where(z==z.max())[0]},
        Tilt: {np.where(z==z.max())[1]}''')

    plt.show()

choice_month(df, 6)
