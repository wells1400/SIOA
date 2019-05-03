import numpy as np
from matplotlib import pyplot as plt


class CoordinateProcess:
    def __init__(self, num_cities=20, min_coord=1, max_coord=5):
        self.num_cities = num_cities  # 要创建的坐标点的数量
        self.min_coord = min_coord  # 最大坐标范围
        self.max_coord = max_coord  # 最小坐标范围

        self.coordinate = self.generate_coordnte()  # 生成随机坐标点
        self.distance_matrix = self.calculate_distance_matrix()  # 各坐标点之间的距离矩阵
        # self.plot_cities()  # 将生成的随机坐标点以散点图的形式绘制出来

    def generate_coordnte(self):
        return np.random.randint(self.min_coord, self.max_coord, size=(self.num_cities, 2))

    def calculate_distance_matrix(self):
        d_mtrx = np.zeros((self.coordinate.shape[0], self.coordinate.shape[0]))
        for indexi in range(d_mtrx.shape[0]):
            for indexj in range(d_mtrx.shape[0]):
                if indexi == indexj:
                    continue
                d_mtrx[indexi][indexj] = np.sqrt(np.power(self.coordinate[indexi] -
                                                          self.coordinate[indexj], 2).sum())
        return d_mtrx

    def plot_cities(self):
        plt.title("City Coordinate")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.coordinate[:, 0], self.coordinate[:, 1])
        for i in range(self.coordinate.shape[0]):
            plt.annotate(str(i), xy=(self.coordinate[i][0], self.coordinate[i][1]))
        plt.show()