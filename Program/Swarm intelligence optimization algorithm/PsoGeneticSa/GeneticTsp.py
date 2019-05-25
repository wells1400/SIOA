import numpy as np
import sys
from random import uniform
from matplotlib import pyplot as plt


# 遗传算法
class GeneticTsp:
    def __init__(self, city_coord, distance_matrix, chrmsome_size=20, cross_prob=0.6, mutate_prob=0.6, iter_round=100,
                 mutate_percentage=0.2
                 ):
        self.city_coord = city_coord
        self.distance_matrix = distance_matrix
        self.chrmsome_size = chrmsome_size
        self.iter_round = iter_round

        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.mutate_percentage = mutate_percentage

        self.individual_list = [self.encode() for _ in range(self.chrmsome_size)]
        self.individual_fitness_list = self.evaluate_fitness(self.individual_list)

        self.new_individual_list = []
        self.new_individual_fitness_list = []

        self.best_chrmsome = self.individual_list[np.argmax(self.individual_fitness_list)]
        self.best_plength = self.__evaluate_distance(self.best_chrmsome)
        self.best_fitness = np.max(self.individual_fitness_list)

        self.mean_fitness_iter = []
        self.best_fitness_iter = []
        self.best_solution_iter = []

    def encode(self):
        init_chrmsome = [i for i in range(len(self.city_coord))]
        np.random.shuffle(init_chrmsome)
        return init_chrmsome

    def __evaluate_distance(self, indv_chrsm):
        path_distance = 0
        for index in range(len(indv_chrsm) - 1):
            path_distance += self.distance_matrix[indv_chrsm[index]][indv_chrsm[index + 1]]
        path_distance += self.distance_matrix[indv_chrsm[0]][indv_chrsm[-1]]
        return path_distance

    def evaluate_fitness(self, individual_list):
        res_fitness_list = []
        for index in range(len(individual_list)):
            indv_chrsm = individual_list[index]
            res_fitness_list.append(1 / self.__evaluate_distance(indv_chrsm))
        return res_fitness_list

    def __roulette_selection(self, fitness_list):
        val_list = np.array(fitness_list)
        sumFits = val_list.sum()
        rndPoint = uniform(0, sumFits)
        accumulator = 0.0
        for ind, val in enumerate(val_list):
            accumulator += val
            if accumulator >= rndPoint:
                return ind

    def roulette_selection(self, fitness_list):
        chrsome_p1 = self.__roulette_selection(fitness_list)
        chrsome_p2 = self.__roulette_selection(fitness_list)
        while True:
            if chrsome_p1 != chrsome_p2:
                break
            chrsome_p2 = self.__roulette_selection(fitness_list)
        return chrsome_p1, chrsome_p2

    def stop_control(self, iter_round):
        return iter_round >= self.iter_round

    def cross(self, chrmsome_p1, chrmsome_p2):
        rnd_point = uniform(0, 1)
        if rnd_point > self.cross_prob:

            rand_points = np.random.randint(0, high=len(city_coord), size=2)
            start_pos = rand_points.min()
            end_pos = rand_points.max()
            selected_part_p1 = chrmsome_p1[start_pos:end_pos]
            selected_part_p2 = chrmsome_p2[start_pos:end_pos]

            cp_chrmsome_p1 = chrmsome_p1.copy()
            cp_chrmsome_p2 = chrmsome_p2.copy()

            for index in range(len(cp_chrmsome_p1)):
                if cp_chrmsome_p1[index] in selected_part_p2:
                    chrmsome_p1.remove(cp_chrmsome_p1[index])
                if cp_chrmsome_p2[index] in selected_part_p1:
                    chrmsome_p2.remove(cp_chrmsome_p2[index])
            rand_insert_point = np.random.randint(0, high=len(chrmsome_p1))
            chrmsome_p1 = chrmsome_p1[:rand_insert_point] + selected_part_p2 + chrmsome_p1[rand_insert_point:]
            chrmsome_p2 = chrmsome_p2[:rand_insert_point] + selected_part_p1 + chrmsome_p2[rand_insert_point:]
        return chrmsome_p1, chrmsome_p2

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

    def natural_select(self):
        while True:

            newindividual_highest_fitness_index = np.argmax(self.new_individual_fitness_list)

            individual_lowest_fitness_index = np.argmin(self.individual_fitness_list)
            if self.new_individual_fitness_list[newindividual_highest_fitness_index] > self.individual_fitness_list[
                individual_lowest_fitness_index]:

                self.individual_list[individual_lowest_fitness_index] = self.new_individual_list[
                    newindividual_highest_fitness_index]
                self.individual_fitness_list[individual_lowest_fitness_index] = self.new_individual_fitness_list[
                    newindividual_highest_fitness_index]

                self.new_individual_fitness_list.pop(newindividual_highest_fitness_index)
                self.new_individual_list.pop(newindividual_highest_fitness_index)
            else:
                break
        return

    def refresh_best_chromosome(self):
        best_index = np.argmax(self.individual_fitness_list)
        self.best_chrmsome = self.individual_list[best_index]
        self.best_fitness = self.individual_fitness_list[best_index]
        self.best_plength = self.__evaluate_distance(self.best_chrmsome)

    def evolve(self):
        pos_generation = 0

        self.new_individual_list = []
        self.new_individual_fitness_list = []
        while True:

            chrmsome_index_1, chrmsome_index_2 = self.roulette_selection(self.individual_fitness_list)

            new_individual_p1, new_individual_p2 = self.cross(self.individual_list[chrmsome_index_1].copy(),
                                                              self.individual_list[chrmsome_index_2].copy())

            new_individual_p1 = self.mutate(new_individual_p1)
            new_individual_p2 = self.mutate(new_individual_p2)

            self.new_individual_list.extend([new_individual_p1, new_individual_p2])
            if pos_generation > self.chrmsome_size / 2:
                break
            pos_generation += 2

        self.new_individual_fitness_list = self.evaluate_fitness(self.new_individual_list)

        self.natural_select()

        self.refresh_best_chromosome()

    def plot_iter(self, pic_save_dir):
        x = [iter_i for iter_i in range(1, len(self.mean_fitness_iter) + 1)]
        y_list = [self.mean_fitness_iter, self.best_fitness_iter, self.best_solution_iter]
        y_lable_list = ['mean_fitness', 'best_fitness', 'shortest_length']
        for y_index in range(len(y_list)):
            plt.title(y_lable_list[y_index] + "_Genetic")
            plt.plot(x, y_list[y_index], label=y_lable_list[y_index])
            plt.xlabel(r'iter_round')
            plt.ylabel(y_lable_list[y_index])
            plt.savefig(pic_save_dir + r'\\' + y_lable_list[y_index] + '.png')
            plt.show()

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
        plt.savefig(pic_save_dir + r'\\' + 'Genetic_path' + '.png')
        plt.show()

    def ga_engine(self, pic_save_dir):
        iter_round = 0
        while True:
            if not self.stop_control(iter_round):
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
