import numpy as np
import sys
from matplotlib import pyplot as plt


# PSO算法
class PsoTsp:
    def __init__(self, city_coord, distance_matrix, partical_size=50, iter_round=1000,
                 alpha=0.5, beta=0.5):
        self.city_coord = city_coord
        self.distance_matrix = distance_matrix

        self.partical_size = partical_size
        self.iter_round = iter_round
        self.var_alpha = alpha
        self.var_beta = beta

        self.partical_pos = [self.__particle_init() for _ in range(self.partical_size)]
        self.partical_vel = []
        self.partical_fitness = self.evaluate_fitness(self.partical_pos)

        self.partical_pos_pbest = self.partical_pos.copy()
        self.partical_fitness_pbest = self.partical_fitness.copy()

        self.partical_pos_gbest = self.partical_pos[np.argmax(self.partical_fitness)]
        self.partical_fitness_gbest = self.__evaluate_fitness(self.partical_pos_gbest)
        self.partical_length_gbest = self.__evaluate_distance(self.partical_pos_gbest)

        self.sorted_edge = self.cal_sorted_edge()

        self.mean_fitness_iter = []
        self.best_fitness_iter = []
        self.best_route_length_iter = []


    def __particle_init(self):
        init_particle = [i for i in range(len(self.city_coord))]
        np.random.shuffle(init_particle)
        return init_particle


    def __evaluate_distance(self, particle_pos):
        path_distance = 0
        for index in range(len(particle_pos) - 1):
            path_distance += self.distance_matrix[particle_pos[index]][particle_pos[index + 1]]
        path_distance += self.distance_matrix[particle_pos[0]][particle_pos[-1]]
        return path_distance


    def __evaluate_fitness(self, particle_pos):
        return 1 / self.__evaluate_distance(particle_pos)


    def evaluate_fitness(self, particle_list):
        res_fitness_list = []
        for index in range(len(particle_list)):
            res_fitness_list.append(self.__evaluate_fitness(particle_list[index]))
        return res_fitness_list


    def stop_control(self, iter_round_counter):
        return iter_round_counter >= self.iter_round


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


    def cal_m(self, iter_round):
        N = len(self.city_coord)
        return round((N - 1) - (N - 5) * iter_round / self.iter_round)


    def _particle_minus_method(self, p_best_pos, particle_pos):
        p_best_pos_edge = [[p_best_pos[index], p_best_pos[index + 1]] for index in range(0, len(p_best_pos) - 1)]
        particle_pos_edge = [set([particle_pos[index], particle_pos[index + 1]]) for index in
                             range(0, len(particle_pos) - 1)]
        res_list = []
        for val in p_best_pos_edge:
            if set(val) not in particle_pos_edge:
                res_list.append(val)
        return res_list


    def _particle_multiply_method(self, para_prob, particle_vel_edge):
        res_list = []
        for index in range(len(particle_vel_edge)):
            rand_point = np.random.rand()
            if rand_point >= para_prob:
                res_list.append(particle_vel_edge[index])
        return res_list


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


    def update_particle_pos(self):
        rand_r = np.random.rand(5)
        for particle_index in range(self.partical_size):
            self._update_particle_pos(rand_r, particle_index)


    def update_particle_vel(self, m):
        self.partical_vel = []

        short_edge_lib = np.array(self.sorted_edge)[:, :m]
        for _ in range(self.partical_size):

            num_edge = np.random.randint(0, high=len(short_edge_lib) * m)

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


    def update_pb_gb(self):
        self.partical_fitness = self.evaluate_fitness(self.partical_pos)
        for particle_index in range(self.partical_size):

            if self.partical_fitness[particle_index] > self.partical_fitness_pbest[particle_index]:
                self.partical_fitness_pbest[particle_index] = self.partical_fitness[particle_index]
                self.partical_pos_pbest[particle_index] = self.partical_pos[particle_index]
        if np.max(self.partical_fitness_pbest) > self.partical_fitness_gbest:

            self.partical_fitness_gbest = np.max(self.partical_fitness_pbest)
            self.partical_pos_gbest = self.partical_pos_pbest[np.argmax(self.partical_fitness_pbest)]
            self.partical_length_gbest = self.__evaluate_distance(self.partical_pos_gbest)


    def iter_record(self):
        self.mean_fitness_iter.append(np.mean(self.partical_fitness_pbest))
        self.best_fitness_iter.append(self.partical_fitness_gbest)
        self.best_route_length_iter.append(self.partical_length_gbest)


    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.mean_fitness_iter) + 1)]
        y_list = [self.mean_fitness_iter, self.best_fitness_iter, self.best_route_length_iter]
        y_lable_list = ['mean_fitness_Pso', 'best_fitness_Pso', 'shortest_length_Pso']
        for y_index in range(len(y_list)):
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\\' + y_lable_list[y_index] + '.png')
            plt.show()


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
        plt.savefig(pic_save_dir + r'\PSO.png')
        plt.show()


    def pso_engine(self, pic_save_dir):
        iter_round = 0
        while not self.stop_control(iter_round):

            m = self.cal_m(iter_round)

            self.update_particle_vel(m)

            self.update_particle_pos()

            self.update_pb_gb()

            self.iter_record()
            iter_round += 1
        self.partical_pos_gbest.append(self.partical_pos_gbest[0])
        print('PSO:')
        print('best path:', self.partical_pos_gbest)
        print('length of best path:', self.partical_length_gbest)
        print('best path fitness:', self.partical_fitness_gbest)
        return self.plot_iter(pic_save_dir), self.plot_path(pic_save_dir)

