import numpy as np
from matplotlib import pyplot as plt


class CoordinateProcess:
    def __init__(self, num_cities=20, min_coord=1, max_coord=5):
        self.num_cities = num_cities  # 要创建的坐标点的数量
        self.min_coord = min_coord  # 最大坐标范围
        self.max_coord = max_coord  # 最小坐标范围

        self.coordinate = self.generate_coordnte()  # 生成随机坐标点
        self.distance_matrix = self.calculate_diantance_matrix()  # 各坐标点之间的距离矩阵
        #self.plot_cities()  # 将生成的随机坐标点以散点图的形式绘制出来

    def generate_coordnte(self):
        return np.random.randint(self.min_coord, self.max_coord, size=(self.num_cities, 2))

    def calculate_distance_dot(self, dot1, dot2):  # 计算两个坐标点之间的位置
        return np.sqrt(np.power(dot1 - dot2, 2).sum())

    def calculate_diantance_matrix(self):
        d_mtrx = np.zeros((self.coordinate.shape[0], self.coordinate.shape[0]))
        for indexi in range(d_mtrx.shape[0]):
            for indexj in range(d_mtrx.shape[0]):
                if indexi == indexj:
                    continue
                d_mtrx[indexi][indexj] = self.calculate_distance_dot(self.coordinate[indexi], self.coordinate[indexj])
        return d_mtrx

    def plot_cities(self):
        plt.title("City Coordinate")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.coordinate[:, 0], self.coordinate[:, 1])
        plt.show()


class AntColony:
    def __init__(self,coordinate_matrix, distance_matrix, ant_size=50, info_alpha=1, heu_beta=5, phe_decay=0.1, phe_amount=1, max_iter=100):
        self.coordinate_matrix = coordinate_matrix  # 坐标矩阵
        self.distance_matrix = distance_matrix  # 距离矩阵

        self.ant_size = ant_size  # 蚁群大小
        self.info_alpha = info_alpha  # 信息素重要度因子
        self.heu_beta = heu_beta  # 启发函数重要度因子
        self.phe_decay = phe_decay  # 信息素衰减系数
        self.phe_amount = phe_amount  # 蚂蚁信息素散布量

        self.max_iter = max_iter  # 最大迭代次数




if __name__ == '__main__':
    coordinateprocessor = CoordinateProcess(10, 1, 10)
    coordinate_matrix = coordinateprocessor.coordinate
    distance_matrix = coordinateprocessor.distance_matrix
    antcolony = AntColony(coordinate_matrix, distance_matrix)



