import numpy as np
import sys
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
    def __init__(self, num_cities=10, min_coord=1, max_coord=20, ant_size=8, info_alpha=1,
                 heu_beta=5, phe_decay=0.1, phe_amount=1, max_iter=100):
        self.coordinate_generator = CoordinateProcess(num_cities, min_coord, max_coord)  # 城市坐标产生器
        self.num_cities = num_cities

        self.coordinate_matrix = self.coordinate_generator.coordinate  # 坐标矩阵
        self.distance_matrix = self.coordinate_generator.distance_matrix  # 距离矩阵
        self.max_iter = max_iter  # 最大迭代次数

        self.ant_size = ant_size  # 蚁群大小
        self.info_alpha = info_alpha  # 信息素重要度因子
        self.phe_beta = heu_beta  # 启发函数重要度因子
        self.phe_decay = phe_decay  # 信息素衰减系数
        self.phe_amount = phe_amount  # 蚂蚁信息素散布量

        self.city_index = np.array([i for i in range(self.num_cities)])  # 城市索引
        self.phe_matrix = np.ones(self.distance_matrix.shape)  # 道路残留信息素矩阵,初始化为1

        self.ant_pos = np.array([np.random.randint(0, self.num_cities) for _ in range(self.ant_size)])  # 蚂蚁初始位置
        self.ant_path_collector = np.array([[self.ant_pos[ant_index]] for ant_index in range(self.ant_size)])  # 蚂蚁路径记录器
        self.ant_path_length_collector = np.array([0 for _ in range(self.ant_size)])  # 蚂蚁路径长度记录表

        self.city_allow = []  # 城市禁忌表
        self.ant_visit_prob = []  # 存储所有蚂蚁访问各城市的概率，均值初始化

        self.best_path = []  # 最优路径
        self.best_path_length = sys.maxsize  # 最优路径的长度

    def ant_init(self):  # 初始化蚁群各蚂蚁的位置、路径记录器、初始禁忌表、城市访问概率表
        self.ant_pos = np.array([np.random.randint(0, self.num_cities) for _ in range(self.ant_size)])  # 蚂蚁所在位置
        self.ant_path_collector = np.array([[self.ant_pos[ant_index]] for ant_index in range(self.ant_size)])  # 蚂蚁路径记录器
        self.ant_path_length_collector = np.array([0 for _ in range(self.ant_size)])  # 蚂蚁路径长度记录表

        self.city_allow = np.array([[city_index for city_index in self.city_index
                                     if city_index != self.ant_pos[i]] for i in range(self.ant_size)])  # 城市禁忌表

        self.ant_visit_prob = np.ones(self.city_allow.shape) / self.city_allow.shape[1]  # 存储所有蚂蚁访问各城市的概率，均值初始化

    def calculate_path_length(self):  # 计算蚂蚁路径的长度
        pass

    def refresh_best(self):  # 更新最优路径和最优路径的长度
        pass

    def ant_phe_distribute(self):  #
        pass

    def ant_best(self):  # 返回最优解并进行可视化
        pass

    def refresh_road_phe(self, ant_system_name='acs'):  # 道路残留信息素矩阵更新
        pass

    def calculate_visitprob(self, ant_index, city_from, city_to):  # 计算蚂蚁ant_index从城市city_from到城市city_to的概率
        top = np.power(self.phe_matrix[city_from][city_to], self.info_alpha) * \
              np.power(1/self.distance_matrix[city_from][city_to], self.phe_beta)
        down = np.power()
        return
        pass

    def roulette_selection(self, ant_index):  # 轮盘对赌选择法确定蚂蚁去往哪个城市
        '''
        :param ant_index: 输入蚂蚁index
        :return: 城市index
        '''
        sum_fits = self.ant_visit_prob[ant_index].sum()
        rnd_point = np.random.uniform(0, sum_fits)
        accumulator = 0.0
        for index, val in enumerate(self.ant_visit_prob[ant_index]):
            accumulator += val
            if accumulator >= rnd_point:
                return self.city_allow[ant_index][index]

    def refresh_ant_visitprob(self):  # 更新各蚂蚁访问各道路的概率
        pass

    def ant_visit(self):  # 产生解空间
        '''
        对每一个蚂蚁按照轮盘对赌选择确定下一个要访问的城市，
        确定好后将该城市从allow中去除并将该城市添加至足迹记录表中，
        直到allow表为空即所有蚂蚁都完成了对所有城市的一次访问。
        :return:
        '''

        pass

    def stop_control(self, iter_round):  # 算法终止控制
        return iter_round >= self.max_iter

    def ant_engine(self):  # 算法运行主程序
        iter_round = 0
        self.ant_init()
        while True:
            if not self.stop_control(iter_round):
                break

        return self.ant_best()


if __name__ == '__main__':
    coordinateprocessor = CoordinateProcess(10, 1, 10)
    coordinate_matrix = coordinateprocessor.coordinate
    distance_matrix = coordinateprocessor.distance_matrix
    antcolony = AntColony(coordinate_matrix, distance_matrix)
    print(np.ones((2,3)))



