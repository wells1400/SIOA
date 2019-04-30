import numpy as np
import sys
from matplotlib import pyplot as plt


class CoordinateProcess:
    def __init__(self, num_cities=20, min_coord=1, max_coord=5):
        self.num_cities = num_cities  # 要创建的坐标点的数量
        self.min_coord = min_coord  # 最大坐标范围
        self.max_coord = max_coord  # 最小坐标范围

        self.coordinate = self.generate_coordnte()  # 生成随机坐标点
        self.distance_matrix = self.calculate_distance_matrix()  # 各坐标点之间的距离矩阵
        self.plot_cities()  # 将生成的随机坐标点以散点图的形式绘制出来

    def generate_coordnte(self):
        return np.random.randint(self.min_coord, self.max_coord, size=(self.num_cities, 2))

    def calculate_distance_matrix(self):
        d_mtrx = np.zeros((self.coordinate.shape[0], self.coordinate.shape[0]))
        for indexi in range(d_mtrx.shape[0]):
            for indexj in range(d_mtrx.shape[0]):
                if indexi == indexj:
                    continue
                d_mtrx[indexi][indexj] = np.sqrt(np.power(self.coordinate[indexi] - self.coordinate[indexj], 2).sum())
        return d_mtrx

    def plot_cities(self):
        plt.title("City Coordinate")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.coordinate[:, 0], self.coordinate[:, 1])
        plt.show()


class AntColony:
    def __init__(self, num_cities=10, min_coord=1, max_coord=20, ant_size=8, info_alpha=1,
                 heu_beta=5, phe_decay=0.1, phe_amount=1, max_iter=100, ant_model='acs'):
        self.coordinate_generator = CoordinateProcess(num_cities, min_coord, max_coord)  # 城市坐标产生器
        self.num_cities = num_cities
        self.city_index = np.array([i for i in range(self.num_cities)])  # 城市索引

        self.coordinate_matrix = self.coordinate_generator.coordinate  # 坐标矩阵
        self.distance_matrix = self.coordinate_generator.distance_matrix  # 距离矩阵
        self.max_iter = max_iter  # 最大迭代次数

        self.ant_size = ant_size  # 蚁群大小
        self.info_alpha = info_alpha  # 信息素重要度因子
        self.phe_beta = heu_beta  # 启发函数重要度因子
        self.phe_decay = phe_decay  # 信息素衰减系数
        self.phe_amount = phe_amount  # 蚂蚁信息素散布量
        self.ant_model = ant_model

        self.phe_matrix = np.ones(self.distance_matrix.shape)  # 道路残留信息素矩阵,初始化为1

        self.ant_pos = np.array([])  # 蚂蚁初始位置
        self.ant_path_collector = []  # 蚂蚁路径记录器
        self.ant_path_length_collector = np.array([])  # 蚂蚁路径长度记录表
        self.city_allow = []  # 城市禁忌表
        self.ant_visit_prob_matrix = np.array([])  # 存储所有蚂蚁访问各城市的概率

        self.best_path = []  # 最优路径
        self.best_path_length = sys.maxsize  # 最优路径的长度

        self.mean_len_record = []
        self.best_len_record = []

    def ant_init(self):  # 初始化蚁群各蚂蚁的位置、路径记录器、初始禁忌表、城市访问概率表
        self.ant_pos = np.array(
            [self.city_index[np.random.randint(0, self.num_cities)] for _ in range(self.ant_size)])  # 蚂蚁所在位置
        self.ant_path_collector = [[self.ant_pos[ant_index]] for ant_index in range(self.ant_size)]  # 蚂蚁路径记录器
        self.ant_path_length_collector = np.array([])  # 蚂蚁路径长度记录表

        self.city_allow = [[city_index for city_index in self.city_index
                            if city_index != self.ant_pos[i]] for i in range(self.ant_size)]  # 城市禁忌表
        self.ant_visit_prob_matrix = self.update_visit_prob_matrix()  # 存储所有蚂蚁访问各城市的概率，均值初始化

    def update_path_length(self):  # 计算蚂蚁路径的长度
        self.ant_path_length_collector = np.array([np.array([self.distance_matrix[
                                                                 self.ant_path_collector[ant_index][dotindex]][
                                                                 self.ant_path_collector[ant_index][dotindex + 1]]
                                                             for dotindex in
                                                             range(len(self.ant_path_collector[ant_index]) - 1)]).sum()
                                                   for ant_index in range(self.ant_size)])

    def update_best(self):  # 更新最优路径和最优路径的长度
        if self.best_path_length > self.ant_path_length_collector.max():
            self.best_path_length = self.ant_path_length_collector.max()
            self.best_path = self.ant_path_collector[np.argmax(self.ant_path_length_collector)]

    def delta_phe(self, ant_model='acs'):
        delta_phe_matrix = np.zeros(shape=self.phe_matrix.shape) * self.phe_amount
        for ant_index in range(self.ant_size):
            for path_index in range(len(self.ant_path_collector) - 1):
                from_city = self.ant_path_collector[path_index]
                to_city = self.ant_path_collector[path_index + 1]
                delta_phe_matrix[from_city][to_city] += self.phe_amount / \
                                                        self.ant_path_length_collector[ant_index]
        return delta_phe_matrix

    def ant_phe_distribute(self, ant_model):  # 三种蚁群模型 acs, aqs, ads 的信息素散布模型
        delta_phe_matrix = np.zeros(shape=self.phe_matrix.shape) * self.phe_amount
        for ant_index in range(self.ant_size):
            if ant_model == 'acs':
                for path_index in range(len(self.ant_path_collector) - 1):
                    from_city = self.ant_path_collector[path_index]
                    to_city = self.ant_path_collector[path_index + 1]
                    delta_phe_matrix[from_city][to_city] += self.phe_amount / \
                                                            self.ant_path_length_collector[ant_index]
            else:
                from_city = self.ant_path_collector[-1]
                to_city = self.ant_path_collector[-2]
                if ant_model == 'aqs':
                    delta_phe_matrix[from_city][to_city] += self.phe_amount / \
                                                            self.distance_matrix[from_city][to_city]
                else:
                    delta_phe_matrix[from_city][to_city] += self.phe_amount
        return delta_phe_matrix

    def ant_best(self):  # 返回最优解并进行可视化
        self.plot_iter_info()
        return self.best_path, self.best_path_length

    def update_road_phe(self, ant_model='acs'):  # 道路残留信息素矩阵更新
        self.phe_matrix += (1 - self.phe_decay) * self.phe_matrix + self.ant_phe_distribute(ant_model)

    def update_visit_prob_matrix(self):  # 更新各蚂蚁访问各道路的概率
        return np.array([[self.calculate_visitprob(ant_index, self.ant_pos[ant_index], city_to) for city_to in
                          self.city_allow[ant_index]]
                         for ant_index in range(self.ant_size)])

    def calculate_visitprob(self, ant_index, city_from, city_to):  # 计算蚂蚁ant_index从城市city_from到城市city_to的概率
        top = np.power(self.phe_matrix[city_from][city_to], self.info_alpha) * \
              np.power(1 / self.distance_matrix[city_from][city_to], self.phe_beta)
        down = np.array([np.power(self.phe_matrix[city_from][city_index], self.info_alpha) *
                         np.power(1 / self.distance_matrix[city_from][city_index], self.phe_beta)
                         for city_index in self.city_allow[ant_index]]).sum()
        return top / down

    def roulette_selection(self, ant_index):  # 轮盘对赌选择法确定蚂蚁去往哪个城市
        '''
        :param ant_index: 输入蚂蚁index
        :return: 城市index
        '''
        sum_fits = self.ant_visit_prob_matrix[ant_index].sum()
        rnd_point = np.random.uniform(0, sum_fits)
        accumulator = 0.0
        for index, val in enumerate(self.ant_visit_prob_matrix[ant_index]):
            accumulator += val
            if accumulator >= rnd_point:
                return self.city_allow[ant_index][index], index

    def ant_visit(self):  # 产生解空间
        '''
        对每一个蚂蚁按照轮盘对赌选择确定下一个要访问的城市，
        确定好后将该城市从allow中去除并将该城市添加至足迹记录表中，
        直到allow表为空即所有蚂蚁都完成了对所有城市的一次访问。
        :return:
        '''
        while True:
            exit_flag = 1
            for ant_index in range(self.ant_size):
                if len(self.ant_path_collector[ant_index]) < self.num_cities:
                    exit_flag = 0
                    # 轮盘选择要访问的城市
                    to_city_index, city_prob_index = self.roulette_selection(ant_index)
                    # 更新目前蚂蚁所在位置,将该城市加入已访问路径表
                    self.ant_pos[ant_index] = to_city_index
                    self.ant_path_collector[ant_index].append(to_city_index)
                    # 从禁忌表中去除当前城市
                    self.city_allow[ant_index][city_prob_index] = -1
                    # 更新访问概率表，将这个蚂蚁的访问概率表中该城市的值设为0
                    self.ant_visit_prob_matrix[ant_index][city_prob_index] = 0
                    # 更新蚁群已经走过的路径长度
                    self.update_path_length()
                    # 如果蚁群模型是 aqs 或者 ads 则所有蚂蚁搜索一步就更新一次道路
                    if self.ant_model == 'aqs' or self.ant_model == 'ads':
                        self.update_road_phe(self.ant_model)
                    continue
                break
            if exit_flag == 1:
                break

    def stop_control(self, iter_round):  # 算法终止控制
        return iter_round >= self.max_iter

    def plot_iter_info(self):
        '''
        :return: 迭代过程个体最优均值和全局最优指标绘制
        '''
        x = [iter_i for iter_i in range(1, self.max_iter + 1)]
        y_list = [self.mean_len_record, self.best_len_record]
        y_lable_list = ['mean_path_length', 'best_length']
        for y_index in range(len(y_list)):
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.show()

    def ant_engine(self):  # 算法运行主程序
        iter_round = 0
        self.ant_init()
        while True:
            if not self.stop_control(iter_round):
                # 产生解空间
                self.ant_visit()
                # 更新最优解
                self.update_best()
                # 更新道路上的信息素（如果是acs 模型）
                if self.ant_model == 'acs':
                    self.update_road_phe(self.ant_model)

                self.mean_len_record.append(self.ant_path_length_collector.mean())
                self.best_len_record.append(self.best_path_length)
                iter_round += 1
            else:
                break
        return self.ant_best()


if __name__ == '__main__':
    coordinateprocessor = CoordinateProcess(10, 1, 10)
    coordinate_matrix = coordinateprocessor.coordinate
    distance_matrix = coordinateprocessor.distance_matrix
    antcolony = AntColony(coordinate_matrix, distance_matrix)
    print(np.ones((2,3)))



