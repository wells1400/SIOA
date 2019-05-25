import numpy as np
from matplotlib import pyplot as plt


# 模拟退火算法
class SimulateAnneal:
    def __init__(self, city_coord, distance_matrix, min_temperature=1, max_temperature=100,
                 cooling_alpha=0.9, iter_round=1000, new_solution_method=0
                 ):
        self.city_coord = city_coord
        self.distance_matrix = distance_matrix

        self.temperature_pos = max_temperature
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature

        self.new_solution_method = new_solution_method
        self.cooling_alpha = cooling_alpha
        self.iter_round = iter_round

        self.solution_local = [i for i in range(len(self.city_coord))]

        self.iter_round_local = 0

        self.best_path_solution = self.solution_local.copy()
        self.best_path_length = self.__evaluate_distance(self.best_path_solution)

        self.inner_loop_round = 100

        self.best_length_iter = []


    def __evaluate_distance(self, solution_local):
        path_distance = 0
        for index in range(len(solution_local) - 1):
            path_distance += self.distance_matrix[solution_local[index]][solution_local[index + 1]]
        path_distance += self.distance_matrix[solution_local[0]][solution_local[-1]]
        return path_distance


    def stop_control(self):
        return self.iter_round_local >= self.iter_round or self.temperature_pos <= self.min_temperature


    def __chose_points(self):
        points_index_list = np.random.randint(0, high=len(self.solution_local), size=2)
        while True:
            if points_index_list[0] != points_index_list[1]:
                break
            points_index_list = np.random.randint(0, high=len(self.solution_local), size=2)
        return points_index_list

    def __generate_solution(self):
        new_solution = self.solution_local.copy()
        if self.new_solution_method == 0:

            points_index_list = self.__chose_points()
            point_index0 = points_index_list[0]
            point_index1 = points_index_list[1]
            new_solution[point_index0], new_solution[point_index1] = \
                new_solution[point_index1], new_solution[point_index0]
        elif self.new_solution_method == 1:

            points_index_list = self.__chose_points()
            pos_start = points_index_list[0] if points_index_list[1] > points_index_list[0] else points_index_list[1]
            pos_end = points_index_list[1] if points_index_list[0] == pos_start else points_index_list[0]
            if pos_start + 1 == pos_end:
                return new_solution
            tmp_part = new_solution[pos_start + 1:pos_end]
            tmp_part.reverse()
            new_solution[pos_start + 1:pos_end] = tmp_part
        return new_solution


    def _update_best_solution(self, new_solution, new_path_length):
        if new_path_length < self.best_path_length:
            self.best_path_solution = new_solution
            self.best_path_length = new_path_length


    def inner_loop(self):
        for _ in range(self.inner_loop_round):
            if self.stop_control():
                break

            new_solution = self.__generate_solution()
            new_path_length = self.__evaluate_distance(new_solution)

            solution_local_length = self.__evaluate_distance(self.solution_local)
            dE = new_path_length - solution_local_length


            if dE <= 0:

                self.solution_local = new_solution
                self._update_best_solution(new_solution, new_path_length)
            else:

                rand_point = np.random.rand()
                if rand_point < np.exp(-dE / self.temperature_pos):
                    self.solution_local = new_solution
            self.iter_round_local += 1
            self.best_length_iter.append(self.best_path_length)


    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.best_length_iter) + 1)]
        y_list = [self.best_length_iter]
        y_lable_list = ['shortest_length_SimulatedAnneal']
        for y_index in range(len(y_list)):
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\SimulatedAnneal_iter.png')
            plt.show()


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
        plt.savefig(pic_save_dir + r'\SimulatedAnneal_path.png')
        plt.show()


    def simulate_anneal_tsp_engine(self, pic_save_dir):
        while not self.stop_control():

            self.inner_loop()

            self.temperature_pos = self.temperature_pos * self.cooling_alpha
        self.best_path_solution.append(self.best_path_solution[0])
        print('Simulated Anneal:')
        print('best path:', self.best_path_solution)
        print('length of best path:', self.best_path_length)
        return self.plot_iter(pic_save_dir), self.plot_path(pic_save_dir)
