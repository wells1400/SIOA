import numpy as np
import sys
from random import uniform
from matplotlib import pyplot as plt


# 点坐标生成模块
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

    def plot_cities(self,pic_save_dir):
        plt.title("City Coordinate")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.coordinate[:, 0], self.coordinate[:, 1])
        for i in range(self.coordinate.shape[0]):
            plt.annotate(str(i), xy=(self.coordinate[i][0], self.coordinate[i][1]))
        plt.savefig(pic_save_dir + r'\\city_coordinate.jpg')
        plt.show()


# 遗传算法
class GeneticTsp:
    def __init__(self, city_coord, distance_matrix, chrmsome_size=20, cross_prob=0.6, mutate_prob=0.6, iter_round=100,
                 mutate_percentage=0.2
                 ):
        self.city_coord = city_coord  # 城市坐标
        self.distance_matrix = distance_matrix  # 距离矩阵
        self.chrmsome_size = chrmsome_size  # 染色体条数
        self.iter_round = iter_round  # 迭代进化次数

        self.cross_prob = cross_prob  # 交叉概率，即能产生新后代的概率
        self.mutate_prob = mutate_prob  # 染色体变异概率
        self.mutate_percentage = mutate_percentage  # 染色体基因变异的比例

        self.individual_list = [self.encode() for _ in range(self.chrmsome_size)]  # 种群染色体集合
        self.individual_fitness_list = self.evaluate_fitness(self.individual_list)  # 计算种群中所有个体初始的适应度

        self.new_individual_list = []  # 新一代染色体集合
        self.new_individual_fitness_list = []  # 新一代染色体适应度集合

        self.best_chrmsome = self.individual_list[np.argmax(self.individual_fitness_list)]  # 最优染色体
        self.best_plength = self.__evaluate_distance(self.best_chrmsome)
        self.best_fitness = np.max(self.individual_fitness_list)

        self.mean_fitness_iter = []
        self.best_fitness_iter = []
        self.best_solution_iter = []

    # 产生随机染色体
    def encode(self):
        init_chrmsome = [i for i in range(len(self.city_coord))]
        np.random.shuffle(init_chrmsome)
        return init_chrmsome

    # 计算路径的长度
    def __evaluate_distance(self, indv_chrsm):
        path_distance = 0
        for index in range(len(indv_chrsm) - 1):
            path_distance += self.distance_matrix[indv_chrsm[index]][indv_chrsm[index + 1]]
        path_distance += self.distance_matrix[indv_chrsm[0]][indv_chrsm[-1]]
        return path_distance

    # 计算种群中所有个体的适应度,适应度计算为路径距离的倒数
    def evaluate_fitness(self, individual_list):
        res_fitness_list = []
        for index in range(len(individual_list)):
            indv_chrsm = individual_list[index]
            res_fitness_list.append(1 / self.__evaluate_distance(indv_chrsm))
        return res_fitness_list

    # 轮盘对赌选择一条染色体
    def __roulette_selection(self, fitness_list):
        val_list = np.array(fitness_list)
        sumFits = val_list.sum()
        rndPoint = uniform(0, sumFits)
        accumulator = 0.0
        for ind, val in enumerate(val_list):
            accumulator += val
            if accumulator >= rndPoint:
                return ind

    # 轮盘对赌染色体选择
    def roulette_selection(self, fitness_list):
        chrsome_p1 = self.__roulette_selection(fitness_list)
        chrsome_p2 = self.__roulette_selection(fitness_list)
        while True:
            if chrsome_p1 != chrsome_p2:
                break
            chrsome_p2 = self.__roulette_selection(fitness_list)
        return chrsome_p1, chrsome_p2

    # 停止运行控制
    def stop_control(self, iter_round):
        return iter_round >= self.iter_round

    # 交叉
    def cross(self, chrmsome_p1, chrmsome_p2):
        rnd_point = uniform(0, 1)
        if rnd_point > self.cross_prob:
            # 产生两个随机数，代表染色体截点
            rand_points = np.random.randint(0, high=len(city_coord), size=2)
            start_pos = rand_points.min()
            end_pos = rand_points.max()
            selected_part_p1 = chrmsome_p1[start_pos:end_pos]
            selected_part_p2 = chrmsome_p2[start_pos:end_pos]

            cp_chrmsome_p1 = chrmsome_p1.copy()
            cp_chrmsome_p2 = chrmsome_p2.copy()
            # 清除重复的点
            for index in range(len(cp_chrmsome_p1)):
                if cp_chrmsome_p1[index] in selected_part_p2:
                    chrmsome_p1.remove(cp_chrmsome_p1[index])
                if cp_chrmsome_p2[index] in selected_part_p1:
                    chrmsome_p2.remove(cp_chrmsome_p2[index])
            rand_insert_point = np.random.randint(0, high=len(chrmsome_p1))
            chrmsome_p1 = chrmsome_p1[:rand_insert_point] + selected_part_p2 + chrmsome_p1[rand_insert_point:]
            chrmsome_p2 = chrmsome_p2[:rand_insert_point] + selected_part_p1 + chrmsome_p2[rand_insert_point:]
        return chrmsome_p1, chrmsome_p2

    # 变异
    def mutate(self, chrmsome_p):
        rnd_point = uniform(0, 1)
        mutate_num = round(self.mutate_percentage * len(chrmsome_p))
        if rnd_point >= self.mutate_prob:
            mutate_count = 0
            while True:
                if mutate_count >= mutate_num:
                    break
                trans_points = np.random.randint(0, len(chrmsome_p), size=2)
                start_pos = trans_points.min()
                end_pos = trans_points.max()
                chrmsome_p[start_pos], chrmsome_p[end_pos] = chrmsome_p[end_pos], chrmsome_p[start_pos]
                mutate_count += 1
        return chrmsome_p

    # 自然选择，优胜劣汰
    def natural_select(self):
        while True:
            # 找到新子代中适应度最高的染色体 index
            newindividual_highest_fitness_index = np.argmax(self.new_individual_fitness_list)
            # 找到旧种群中适应度最低的染色体 index
            individual_lowest_fitness_index = np.argmin(self.individual_fitness_list)
            if self.new_individual_fitness_list[newindividual_highest_fitness_index] > self.individual_fitness_list[
                individual_lowest_fitness_index]:
                # 个体替代
                self.individual_list[individual_lowest_fitness_index] = self.new_individual_list[
                    newindividual_highest_fitness_index]
                self.individual_fitness_list[individual_lowest_fitness_index] = self.new_individual_fitness_list[
                    newindividual_highest_fitness_index]
                # 删除新子代中适应度最高的染色体
                self.new_individual_fitness_list.pop(newindividual_highest_fitness_index)
                self.new_individual_list.pop(newindividual_highest_fitness_index)
            else:
                break
        return

    # 更新最优染色体
    def refresh_best_chromosome(self):
        best_index = np.argmax(self.individual_fitness_list)
        self.best_chrmsome = self.individual_list[best_index]
        self.best_fitness = self.individual_fitness_list[best_index]
        self.best_plength = self.__evaluate_distance(self.best_chrmsome)

    #  进行一次进化过程
    def evolve(self):
        pos_generation = 0
        # 初始化子代种群染色体集合
        self.new_individual_list = []
        self.new_individual_fitness_list = []
        while True:
            # 选择两个染色体
            chrmsome_index_1, chrmsome_index_2 = self.roulette_selection(self.individual_fitness_list)
            # 交叉
            new_individual_p1, new_individual_p2 = self.cross(self.individual_list[chrmsome_index_1].copy(),
                                                              self.individual_list[chrmsome_index_2].copy())
            # 变异
            new_individual_p1 = self.mutate(new_individual_p1)
            new_individual_p2 = self.mutate(new_individual_p2)
            # 存放在新个体集合中
            self.new_individual_list.extend([new_individual_p1, new_individual_p2])
            if pos_generation > self.chrmsome_size / 2:
                break
            pos_generation += 2
        # 计算子代染色体适应度
        self.new_individual_fitness_list = self.evaluate_fitness(self.new_individual_list)
        # 更新换代，用新子代最好的去替代旧种群中最差的
        self.natural_select()
        # 更新全局最优个体
        self.refresh_best_chromosome()

    #  绘图
    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.mean_fitness_iter) + 1)]
        y_list = [self.mean_fitness_iter, self.best_fitness_iter, self.best_solution_iter]
        y_lable_list = ['mean_fitness', 'best_fitness', 'shortest_length']
        for y_index in range(len(y_list)):
            plt.title(y_lable_list[y_index] + "_Genetic")
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\\' + y_lable_list[y_index] + '.jpg')
            plt.show()

    # 绘制最优路径图
    def plot_path(self, pic_save_dir):
        plt.title("City Coordinate_Genetic")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.city_coord[:, 0], self.city_coord[:, 1])
        for i in range(self.city_coord.shape[0]):
            plt.annotate(str(i), xy=(self.city_coord[i][0],
                                     self.city_coord[i][1]
                                     ))
        line_x = [self.city_coord[i, 0] for i in self.best_chrmsome]
        line_y = [self.city_coord[i, 1] for i in self.best_chrmsome]
        plt.annotate('S_E', xy=(line_x[0], line_y[0]))
        plt.plot(line_x, line_y)
        plt.savefig(pic_save_dir + r'\\' + 'Genetic_path' + '.jpg')
        plt.show()

    # 主运行程序
    def ga_engine(self, pic_save_dir):
        iter_round = 0
        while True:
            if not self.stop_control(iter_round):
                # 选择两个
                iter_round += 1
                self.evolve()
                self.mean_fitness_iter.append(np.mean(self.individual_fitness_list))
                self.best_fitness_iter.append(self.best_fitness)
                self.best_solution_iter.append(self.best_plength)
                continue
            break
        self.best_chrmsome.append(self.best_chrmsome[0])
        print('Genetic Algorithm:')
        print('best path:', self.best_chrmsome)
        print('length of best path:', self.best_plength)
        print('best path fitness:', self.best_fitness)
        return self.plot_iter(pic_save_dir), self.plot_path(pic_save_dir)


# PSO算法
class PsoTsp:
    def __init__(self, city_coord, distance_matrix, partical_size=50, iter_round=1000,
                 alpha=0.5, beta=0.5):
        self.city_coord = city_coord  # 城市坐标
        self.distance_matrix = distance_matrix  # 距离矩阵

        self.partical_size = partical_size  # 粒子个数
        self.iter_round = iter_round  # 迭代次数
        self.var_alpha = alpha  # 常数系数c1
        self.var_beta = beta  # 常数系数c2

        self.partical_pos = [self.__particle_init() for _ in range(self.partical_size)]  # 粒子群位置列表
        self.partical_vel = []  # 粒子群速度列表
        self.partical_fitness = self.evaluate_fitness(self.partical_pos)  # 粒子群适应度列表

        self.partical_pos_pbest = self.partical_pos.copy()  # 粒子群个体最优位置
        self.partical_fitness_pbest = self.partical_fitness.copy()  # 粒子群个体最优适应度

        self.partical_pos_gbest = self.partical_pos[np.argmax(self.partical_fitness)]  # 粒子群全局最优位置
        self.partical_fitness_gbest = self.__evaluate_fitness(self.partical_pos_gbest)  # 粒子群全局最优适应度
        self.partical_length_gbest = self.__evaluate_distance(self.partical_pos_gbest)

        self.sorted_edge = self.cal_sorted_edge()  # 计算距离排序矩阵

        self.mean_fitness_iter = []  # 记录个体最优适应度均值随迭代次数变化
        self.best_fitness_iter = []  # 记录全局最优适应度随迭代次数变化
        self.best_route_length_iter = []  # 记录全局最优路径的路径长度随迭代次数的变化

    # 生成一个随机粒子
    def __particle_init(self):
        init_particle = [i for i in range(len(self.city_coord))]
        np.random.shuffle(init_particle)
        return init_particle

    # 计算粒子位置所代表的路径距离
    def __evaluate_distance(self, particle_pos):
        path_distance = 0
        for index in range(len(particle_pos) - 1):
            path_distance += self.distance_matrix[particle_pos[index]][particle_pos[index + 1]]
        path_distance += self.distance_matrix[particle_pos[0]][particle_pos[-1]]
        return path_distance

    # 计算粒子适应度
    def __evaluate_fitness(self, particle_pos):
        return 1 / self.__evaluate_distance(particle_pos)

    # 计算粒子群的适应度
    def evaluate_fitness(self, particle_list):
        res_fitness_list = []
        for index in range(len(particle_list)):
            res_fitness_list.append(self.__evaluate_fitness(particle_list[index]))
        return res_fitness_list

    #  停止控制
    def stop_control(self, iter_round_counter):
        '''
        算法迭代终止控制函数
        :param iter_round_counter:
        :return:
        '''
        return iter_round_counter >= self.iter_round

    # 计算距离排序矩阵
    def cal_sorted_edge(self):
        res_list = []
        cp_distance_matrix = self.distance_matrix.copy()
        for edge_index in range(len(cp_distance_matrix)):
            distance_edge = cp_distance_matrix[edge_index]
            distance_edge[edge_index] = sys.maxsize
            tmp_container = []
            while len(tmp_container) < len(distance_matrix) - 1:
                min_distance, min_index = np.min(distance_edge), np.argmin(distance_edge)
                tmp_container.append(min_index)
                distance_edge[min_index] = sys.maxsize
            res_list.append(tmp_container)
        return res_list

    # 计算短边库长度参数m
    def cal_m(self, iter_round):
        N = len(self.city_coord)
        return round((N - 1) - (N - 5) * iter_round / self.iter_round)

    # 粒子离散减法
    def _particle_minus_method(self, p_best_pos, particle_pos):
        p_best_pos_edge = [[p_best_pos[index], p_best_pos[index + 1]] for index in range(0, len(p_best_pos) - 1)]
        particle_pos_edge = [set([particle_pos[index], particle_pos[index + 1]]) for index in
                             range(0, len(particle_pos) - 1)]
        res_list = []
        for val in p_best_pos_edge:
            if set(val) not in particle_pos_edge:
                res_list.append(val)
        return res_list

    # 粒子离散乘法
    def _particle_multiply_method(self, para_prob, particle_vel_edge):
        res_list = []
        for index in range(len(particle_vel_edge)):
            rand_point = np.random.rand()
            if rand_point >= para_prob:
                res_list.append(particle_vel_edge[index])
        return res_list

    # 粒子离散加法
    def _particle_addition_method(self, particle_pos, particle_vel):
        for each_vel in particle_vel:
            pos_1 = particle_pos.index(each_vel[0])
            pos_2 = particle_pos.index(each_vel[1])
            pos_start = pos_2 if pos_1 > pos_2 else pos_1
            pos_end = pos_1 if pos_2 == pos_start else pos_2
            if pos_start + 1 == pos_end:
                continue
            tmp_part = particle_pos[pos_start + 1:pos_end]
            tmp_part.reverse()
            particle_pos[pos_start + 1:pos_end] = tmp_part
        return particle_pos

    # 更新单个粒子位置
    def _update_particle_pos(self, rand_r, particle_pos_index):
        particle_pos = self.partical_pos[particle_pos_index]
        if rand_r[2] < self.var_alpha:
            vel_1 = self._particle_minus_method(self.partical_pos_pbest[particle_pos_index], particle_pos)
            vel_2 = self._particle_multiply_method(rand_r[0], vel_1)
            self.partical_pos[particle_pos_index] = self._particle_addition_method(particle_pos, vel_2)
        elif rand_r[3] < self.var_beta:
            vel_1 = self._particle_minus_method(self.partical_pos_gbest, particle_pos)
            vel_2 = self._particle_multiply_method(rand_r[1], vel_1)
            self.partical_pos[particle_pos_index] = self._particle_addition_method(particle_pos, vel_2)
        else:
            vel = self.partical_vel[particle_pos_index]
            vel_2 = self._particle_multiply_method(rand_r[4], vel)
            self.partical_pos[particle_pos_index] = self._particle_addition_method(particle_pos, vel_2)

    # 更新粒子的位置
    def update_particle_pos(self):
        rand_r = np.random.rand(5)
        for particle_index in range(self.partical_size):
            self._update_particle_pos(rand_r, particle_index)

    # 更新粒子的速度
    def update_particle_vel(self, m):
        self.partical_vel = []
        # 短边库
        short_edge_lib = np.array(self.sorted_edge)[:, :m]
        for _ in range(self.partical_size):
            # 粒子速度中添加边的个数
            num_edge = np.random.randint(0, high=len(short_edge_lib) * m)
            # 随机添加 num_edge 条边
            edge_container = []
            add_count = 0
            while True:
                if add_count >= num_edge:
                    break
                edge_start = np.random.randint(0, len(self.city_coord))
                edge_end = np.random.randint(0, m)
                if [edge_start, short_edge_lib[edge_start][edge_end]] not in edge_container:
                    edge_container.append([edge_start, short_edge_lib[edge_start][edge_end]])
                add_count += 1
            self.partical_vel.append(edge_container)

    # 更新个体最优和全局最优解
    def update_pb_gb(self):
        self.partical_fitness = self.evaluate_fitness(self.partical_pos)
        for particle_index in range(self.partical_size):
            #  更新个体最优位置和适应度
            if self.partical_fitness[particle_index] > self.partical_fitness_pbest[particle_index]:
                self.partical_fitness_pbest[particle_index] = self.partical_fitness[particle_index]
                self.partical_pos_pbest[particle_index] = self.partical_pos[particle_index]
        if np.max(self.partical_fitness_pbest) > self.partical_fitness_gbest:
            #  更新全局最优位置和适应度
            self.partical_fitness_gbest = np.max(self.partical_fitness_pbest)
            self.partical_pos_gbest = self.partical_pos_pbest[np.argmax(self.partical_fitness_pbest)]
            self.partical_length_gbest = self.__evaluate_distance(self.partical_pos_gbest)

    # 记录不同迭代次数下的个体最优适应度均值，全局最优适应度和最优路径的长度
    def iter_record(self):
        self.mean_fitness_iter.append(np.mean(self.partical_fitness_pbest))
        self.best_fitness_iter.append(self.partical_fitness_gbest)
        self.best_route_length_iter.append(self.partical_length_gbest)

    #  绘图
    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.mean_fitness_iter) + 1)]
        y_list = [self.mean_fitness_iter, self.best_fitness_iter, self.best_route_length_iter]
        y_lable_list = ['mean_fitness_Pso', 'best_fitness_Pso', 'shortest_length_Pso']
        for y_index in range(len(y_list)):
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\\' + y_lable_list[y_index] + '.jpg')
            plt.show()

    # 绘制最优路径图
    def plot_path(self, pic_save_dir):
        plt.title("City Coordinate_PSO")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.city_coord[:, 0], self.city_coord[:, 1])
        for i in range(self.city_coord.shape[0]):
            plt.annotate(str(i), xy=(self.city_coord[i][0],
                                     self.city_coord[i][1]
                                     ))
        line_x = [self.city_coord[i, 0] for i in self.partical_pos_gbest]
        line_y = [self.city_coord[i, 1] for i in self.partical_pos_gbest]
        plt.annotate('S_E', xy=(line_x[0], line_y[0]))
        plt.plot(line_x, line_y)
        plt.savefig(pic_save_dir + r'\PSO.jpg')
        plt.show()

    # 迭代主程
    def pso_engine(self, pic_save_dir):
        iter_round = 0
        while not self.stop_control(iter_round):
            # 计算短边库容量
            m = self.cal_m(iter_round)
            # 计算粒子的速度
            self.update_particle_vel(m)
            # 更新粒子的位置
            self.update_particle_pos()
            # 更新个体最优和全局最优解
            self.update_pb_gb()
            # 记录不同迭代次数下的个体最优适应度均值，全局最优适应度和最优路径的长度
            self.iter_record()
            iter_round += 1
        self.partical_pos_gbest.append(self.partical_pos_gbest[0])
        print('PSO:')
        print('best path:', self.partical_pos_gbest)
        print('length of best path:', self.partical_length_gbest)
        print('best path fitness:', self.partical_fitness_gbest)
        return self.plot_iter(pic_save_dir), self.plot_path(pic_save_dir)


# 模拟退火算法
class SimulateAnneal:
    def __init__(self, city_coord, distance_matrix, min_temperature=1, max_temperature=100,
                 cooling_alpha=0.9, iter_round=1000, new_solution_method=0
                 ):
        self.city_coord = city_coord  # 城市坐标
        self.distance_matrix = distance_matrix  # 距离矩阵

        self.temperature_pos = max_temperature  # 当前温度
        self.max_temperature = max_temperature  # 最大温度
        self.min_temperature = min_temperature  # 最低温度

        self.new_solution_method = new_solution_method  # 产生新解的方法
        self.cooling_alpha = cooling_alpha  # 降温系数
        self.iter_round = iter_round  # 最大迭代次数

        self.solution_local = [i for i in range(len(self.city_coord))]  # 初始解

        self.iter_round_local = 0  # 当前迭代次数

        self.best_path_solution = self.solution_local.copy()  # 最优解
        self.best_path_length = self.__evaluate_distance(self.best_path_solution)  # 最优路径距离

        self.inner_loop_round = 100  # 内循环迭代次数

        self.best_length_iter = []

    # 计算当前解所代表的路径距离
    def __evaluate_distance(self, solution_local):
        path_distance = 0
        for index in range(len(solution_local) - 1):
            path_distance += self.distance_matrix[solution_local[index]][solution_local[index + 1]]
        path_distance += self.distance_matrix[solution_local[0]][solution_local[-1]]
        return path_distance

    #  停止控制
    def stop_control(self):
        return self.iter_round_local >= self.iter_round or self.temperature_pos <= self.min_temperature

    # 确定两个不同的节点
    def __chose_points(self):
        points_index_list = np.random.randint(0, high=len(self.solution_local), size=2)
        while True:
            if points_index_list[0] != points_index_list[1]:
                break
            points_index_list = np.random.randint(0, high=len(self.solution_local), size=2)
        return points_index_list

    # 基于邻域关系产生一个新解
    def __generate_solution(self):
        new_solution = self.solution_local.copy()
        if self.new_solution_method == 0:
            #  随机选择2个节点，交换路径中的这2个节点的顺序
            points_index_list = self.__chose_points()
            point_index0 = points_index_list[0]
            point_index1 = points_index_list[1]
            new_solution[point_index0], new_solution[point_index1] = \
                new_solution[point_index1], new_solution[point_index0]
        elif self.new_solution_method == 1:
            # 随机选择2个节点，将路径中这2个节点间的节点顺序逆转
            points_index_list = self.__chose_points()
            pos_start = points_index_list[0] if points_index_list[1] > points_index_list[0] else points_index_list[1]
            pos_end = points_index_list[1] if points_index_list[0] == pos_start else points_index_list[0]
            if pos_start + 1 == pos_end:
                return new_solution
            tmp_part = new_solution[pos_start + 1:pos_end]
            tmp_part.reverse()
            new_solution[pos_start + 1:pos_end] = tmp_part
        return new_solution

    # 更新最优解
    def _update_best_solution(self, new_solution, new_path_length):
        if new_path_length < self.best_path_length:
            self.best_path_solution = new_solution
            self.best_path_length = new_path_length

    #  内循环函数
    def inner_loop(self):
        for _ in range(self.inner_loop_round):
            if self.stop_control():
                break
            # 基于邻域产生一个新解
            new_solution = self.__generate_solution()
            new_path_length = self.__evaluate_distance(new_solution)
            # 旧解路径长度
            solution_local_length = self.__evaluate_distance(self.solution_local)
            dE = new_path_length - solution_local_length

            # 判断新解是否优于当前解
            if dE <= 0:
                # 接受新解并且判断是否更新最优解
                self.solution_local = new_solution
                self._update_best_solution(new_solution, new_path_length)
            else:
                # 一定概率接受新解
                rand_point = np.random.rand()
                if rand_point < np.exp(-dE / self.temperature_pos):
                    self.solution_local = new_solution
            self.iter_round_local += 1
            self.best_length_iter.append(self.best_path_length)

    #  绘图
    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.best_length_iter) + 1)]
        y_list = [self.best_length_iter]
        y_lable_list = ['shortest_length_SimulatedAnneal']
        for y_index in range(len(y_list)):
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\SimulatedAnneal_iter.jpg')
            plt.show()

    # 绘制最优路径图
    def plot_path(self, pic_save_dir):
        plt.title("City Coordinate_Simulated Anneal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.city_coord[:, 0], self.city_coord[:, 1])
        for i in range(self.city_coord.shape[0]):
            plt.annotate(str(i), xy=(self.city_coord[i][0],
                                     self.city_coord[i][1]
                                     ))
        line_x = [self.city_coord[i, 0] for i in self.best_path_solution]
        line_y = [self.city_coord[i, 1] for i in self.best_path_solution]
        plt.annotate('S_E', xy=(line_x[0], line_y[0]))
        plt.plot(line_x, line_y)
        plt.savefig(pic_save_dir + r'\SimulatedAnneal_path.jpg')
        plt.show()

    # 模拟退火函数主程
    def simulate_anneal_tsp_engine(self, pic_save_dir):
        while not self.stop_control():
            # 内循环
            self.inner_loop()
            # 降温
            self.temperature_pos = self.temperature_pos * self.cooling_alpha
        self.best_path_solution.append(self.best_path_solution[0])
        print('Simulated Anneal:')
        print('best path:', self.best_path_solution)
        print('length of best path:', self.best_path_length)
        return self.plot_iter(pic_save_dir), self.plot_path(pic_save_dir)


# 蚁群算法
class AntColony:
    def __init__(self, city_coord, distance_matrix, ant_size=8, info_alpha=1,
                 heu_beta=5, phe_decay=0.1, phe_amount=1, max_iter=100, ant_model='acs'):
        self.city_index = np.array([i for i in range(len(city_coord))])  # 城市索引

        self.city_coord = city_coord  # 坐标矩阵
        self.distance_matrix = distance_matrix  # 距离矩阵
        self.max_iter = max_iter  # 最大迭代次数

        self.ant_size = ant_size  # 蚁群大小
        self.info_alpha = info_alpha  # 信息素重要度因子
        self.phe_beta = heu_beta  # 启发函数重要度因子
        self.phe_decay = phe_decay  # 信息素衰减系数
        self.phe_amount = phe_amount  # 蚂蚁信息素散布量
        self.ant_model = ant_model

        self.phe_matrix = np.ones(self.distance_matrix.shape)  # 道路残留信息素矩阵,初始化为1

        self.ant_pos = []
        self.ant_city_allow = [[]]
        self.ant_path_collector = [[]]
        self.ant_plen_collector = []
        # self.ant_init()

        self.best_path = []
        self.best_path_length = sys.maxsize
        self.mean_len_record = []
        self.best_len_record = []

    def ant_init(self):
        '''
        初始化蚂蚁位置，禁忌表，路径记录表，路径长度表
        :return:
        '''
        self.ant_pos = [self.city_index[np.random.randint(0, len(self.city_coord))] for _ in range(self.ant_size)]
        self.ant_city_allow = [[city_index for city_index in self.city_index if city_index != self.ant_pos[i]] for i in
                               range(self.ant_size)]
        self.ant_path_collector = [[self.ant_pos[i]] for i in range(self.ant_size)]
        self.ant_plen_collector = [0 for _ in range(self.ant_size)]

    def calculate_visitprob(self, ant_index, city_from, city_to):
        '''
        # 计算蚂蚁ant_index从城市city_from到城市city_to的概率,取决于进禁忌表，信息素矩阵以及距离矩阵
        :param ant_index:
        :param city_from:
        :param city_to:
        :return:
        '''

        top = np.power(self.phe_matrix[city_from][city_to], self.info_alpha) * \
              np.power(1 / self.distance_matrix[city_from][city_to], self.phe_beta)
        down = np.array([np.power(self.phe_matrix[city_from][city_index], self.info_alpha) *
                         np.power(1 / self.distance_matrix[city_from][city_index], self.phe_beta)
                         for city_index in self.ant_city_allow[ant_index]]).sum()
        return top / down

    def roulette_selection(self, ant_index):  # 轮盘对赌选择法确定蚂蚁去往哪个城市
        '''
        :param ant_index: 输入蚂蚁index
        :return: 城市index
        '''
        ant_visit_prob = np.array([self.calculate_visitprob(ant_index, self.ant_pos[ant_index], city_to)
                                   for city_to in self.ant_city_allow[ant_index]])
        sum_fits = ant_visit_prob.sum()
        rnd_point = np.random.uniform(0, sum_fits)
        accumulator = 0.0
        for index, val in enumerate(ant_visit_prob):
            accumulator += val
            if accumulator >= rnd_point:
                return self.ant_city_allow[ant_index][index]

    def calculate_delta_phe(self):
        '''
        :return: delta 蚂蚁散布的信息素矩阵
        '''
        phe_dis_matrix = np.zeros(shape=self.phe_matrix.shape)
        if self.ant_model == 'aqs' or self.ant_model == 'ads':
            for ant_index in range(self.ant_size):
                city_from = self.ant_path_collector[ant_index][-2]
                city_to = self.ant_path_collector[ant_index][-1]
                if self.ant_model == 'aqs':
                    phe_dis_matrix[city_from][city_to] += self.phe_amount / self.distance_matrix[city_from][city_to]
                else:
                    phe_dis_matrix[city_from][city_to] += self.phe_amount
        if self.ant_model == 'acs':
            for ant_index in range(self.ant_size):
                dis_phe_amount = self.phe_amount / self.ant_plen_collector[ant_index]
                for path_index in range(len(self.city_coord) - 1):
                    city_from = self.ant_path_collector[ant_index][path_index]
                    city_to = self.ant_path_collector[ant_index][path_index + 1]
                    phe_dis_matrix[city_from][city_to] += dis_phe_amount
        return phe_dis_matrix

    def ant_visit(self):
        '''
        一次搜寻，所有蚂蚁都完成对所有城市的一次访问
        :return:
        '''
        while True:
            if len(self.ant_path_collector[-1]) >= len(self.city_coord):
                break
            else:
                # 所有蚂蚁都选择一个城市
                for ant_index in range(self.ant_size):
                    # 选择一个城市
                    selected_city = self.roulette_selection(ant_index)
                    # 更新位置, 添加路径
                    self.ant_pos[ant_index] = selected_city
                    self.ant_path_collector[ant_index].append(selected_city)
                    # 从禁忌表中删除此城市
                    self.ant_city_allow[ant_index].remove(selected_city)
                if self.ant_model in ['aqs', 'ads']:
                    phe_dis_matrix = self.calculate_delta_phe()
                    #  所有蚂蚁都向前搜索一步之后，更新这些道路上的信息素
                    self.phe_matrix += phe_dis_matrix

    def is_stop(self, iter_round):
        return iter_round >= self.max_iter

    def update_path_length(self):
        '''
        更新路径长度记录
        :return:
        '''
        for ant_index in range(self.ant_size):
            path_length = 0
            for path_index in range(len(self.city_coord) - 1):
                city_from = self.ant_path_collector[ant_index][path_index]
                city_to = self.ant_path_collector[ant_index][path_index + 1]
                path_length += self.distance_matrix[city_from][city_to]
            self.ant_plen_collector[ant_index] = \
                path_length + self.distance_matrix[self.ant_path_collector[ant_index][-1]][
                    self.ant_path_collector[ant_index][0]]

    def update_road_phe(self):
        self.phe_matrix = (1 - self.phe_decay) * self.phe_matrix + self.calculate_delta_phe()

    def update_best(self):
        length_array = np.array(self.ant_plen_collector)
        if self.best_path_length > length_array.min():
            self.best_path_length = length_array.min()
            self.best_path = self.ant_path_collector[np.argmin(length_array)]
            self.best_path.append(self.best_path[0])

    def plot_path(self, pic_save_dir):
        plt.title("City Coordinate_AntColony")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.city_coord[:, 0], self.city_coord[:, 1])
        for i in range(self.city_coord.shape[0]):
            plt.annotate(str(i), xy=(self.city_coord[i][0],
                                     self.city_coord[i][1]
                                     ))
        line_x = [self.city_coord[i, 0] for i in self.best_path]
        line_y = [self.city_coord[i, 1] for i in self.best_path]
        plt.annotate('S_E', xy=(line_x[0], line_y[0]))
        plt.plot(line_x, line_y)
        plt.savefig(pic_save_dir + r'\AntColony_path.jpg')
        plt.show()

    def plot_iter_info(self,pic_save_dir):
        '''
        :return: 迭代过程个体最优均值和全局最优指标绘制
        '''
        x = [iter_i for iter_i in range(1, self.max_iter + 1)]
        y_list = [self.mean_len_record, self.best_len_record]
        y_lable_list = ['mean_path_length', 'best_length']
        for y_index in range(len(y_list)):
            plt.title(y_lable_list[y_index] + '_AntColony')
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\\' + y_lable_list[y_index] + '.jpg')
            plt.show()

    def ant_engine(self, pic_save_dir):  # 主运行程序
        iter_round = 0
        while True:
            if not self.is_stop(iter_round):
                # 初始化蚂蚁位置
                self.ant_init()
                # 创建解空间
                self.ant_visit()
                # 计算路径的长度
                self.update_path_length()
                # 更新最优解
                self.update_best()
                # 更新道路信息素
                self.update_road_phe()
                self.mean_len_record.append(np.array(self.ant_plen_collector).mean())
                self.best_len_record.append(self.best_path_length)
                iter_round += 1
                if iter_round % 10 == 0:
                    print("ant_iter_round:", iter_round)
                continue
            break
        print('Ant Cololny:')
        print('best path:', self.best_path)
        print('length of best path:', self.best_path_length)
        return self.plot_path(pic_save_dir), self.plot_iter_info(pic_save_dir)


if __name__ == '__main__':
    pic_save_dir = r'D:\WORK__wells\Other Program\TSP'

    # 点坐标生成
    coord_generator = CoordinateProcess(num_cities=15, min_coord=1, max_coord=100)
    coord_generator.plot_cities(pic_save_dir)
    city_coord = coord_generator.coordinate
    distance_matrix = coord_generator.distance_matrix

    print('城市点坐标')
    print(city_coord)

    # 遗传算法效果
    print("遗传算法：")
    test_genetic = GeneticTsp(city_coord, distance_matrix, chrmsome_size=100, cross_prob=0.6, mutate_prob=0.6,
                              iter_round=1000,mutate_percentage=0.2)
    test_genetic.ga_engine(pic_save_dir)

    # PSO效果
    print("粒子群算法：")
    test_pso = PsoTsp(city_coord, distance_matrix,
                      partical_size=100, iter_round=1000, alpha=0.7, beta=0.7)
    test_pso.pso_engine(pic_save_dir)

    # 模拟退火效果
    print('模拟退火算法：')
    simu_anneal = SimulateAnneal(city_coord, distance_matrix,
                                 min_temperature=1, max_temperature=100,
                                 cooling_alpha=0.8, iter_round=10000,
                                 new_solution_method=0)
    simu_anneal.simulate_anneal_tsp_engine(pic_save_dir)

    # 蚁群算法
    print('蚁群算法')
    ant_test = AntColony(city_coord, distance_matrix, ant_size=30, info_alpha=1,
                         heu_beta=2, phe_decay=0.1, phe_amount=10, max_iter=100, ant_model='aqs')
    ant_test.ant_engine(pic_save_dir)


