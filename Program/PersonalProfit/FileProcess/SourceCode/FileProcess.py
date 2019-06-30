import time
import os
from multiprocessing import Pool,Process,Lock
import multiprocessing
multiprocessing.freeze_support()


class FileProcess:
    def __init__(self,
                 root_path=r'E:\SIOA\Program\PersonalProfit\FileProcess\SourceData',
                 n_jobs=4
                 ):
        self.root_path = root_path
        # 计数器
        self.iter_count = 0
        self.max_iters = -1
        # 要计算的文件路径组合
        self.path_container = []
        # 最大进程数
        self.n_jobs = n_jobs

    @staticmethod
    def file_process(file_path_1, file_path_2, file_path_3):
        print('Start at:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        base_file_1 = set()
        base_file_2 = set()
        with open(file_path_1, 'r') as file1:
            with open(file_path_2, 'r') as file2:
                for line in file2:
                    line = line.strip()
                    base_file_2.update([line])
                for line in file1:
                    line = line.strip()
                    base_file_1.update([line])
        res_data = list(base_file_1 & base_file_2)
        print('End at:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        with open(file_path_3, 'w') as f:
            for val in res_data:
                f.write(val)
                f.write('\n')

    # 访问路径生成器,返回输入的最大循环次数
    def path_generator(self):
        test_count = int(input(r'请输入要循环测次数：'))
        self.max_iters = test_count
        file_path_1_index = input(r'请输入file_path_1名：')
        file_path_2_index = input(r'请输入file_path_2名：')

        file_path_1 = self.root_path + r'\\' + file_path_1_index + r'\\'
        file_path_2 = self.root_path + r'\\' + file_path_2_index + r'\\'

        out_file_index = 1

        # 读取file_path_1文件夹内所有txt文件
        files_list = os.listdir(file_path_1)
        files_entire_list = [file_path_1 + files_list[i] for i in range(len(files_list))]
        for base_index in range(len(files_list) - 1):
            base_name = files_list[base_index].split(r'.')[0]
            for pos_index in range(base_index + 1, len(files_list)):
                pos_name = files_list[pos_index].split(r'.')[0]
                # file_path_3 = file_path_2 + str(out_file_index) + r'_' + base_name + '_to_' + pos_name + r'.txt'
                file_path_3 = file_path_2 + r'00' + str(out_file_index) + r'.txt'
                self.path_container.append([files_entire_list[base_index], files_entire_list[pos_index], file_path_3])
                out_file_index += 1

    # 多进程-单任务运行程序
    def worker_run(self):
        print('进程ID %s 在运行' % os.getpid())
        while len(self.path_container) > 0:
            items = self.path_container.pop()
            print(r'file_path_1: <<  %s >>'% items[0])
            print(r'file_path_2: <<  %s >>'% items[1])
            print(r'file_path_3: <<  %s >>'% items[2])
            self.file_process(items[0], items[1], items[2])
            self.iter_count += 1
            if self.iter_count >= self.max_iters:
                return

    # 运行主程序-多进程调用
    def run_engine(self):
        self.path_generator()
        for i in range(self.n_jobs):
            p = Process(target=self.worker_run)
            p.start()
            p.join()


if __name__ == '__main__':
    # 备注：：：：：
    # 要手动输入的是要计算的文件所放的文件夹（file_path_1）,以及输出文件需要存放的文件夹（file_path_2）这两个文件夹的名字
    # 以这里的测试数据为例，file_path_1输入是01， file_path_2输入时02
    # 然后root_path是主目录（即01和02文件夹的父文件夹的绝对路径，
    # 我这里是E:\SIOA\Program\PersonalProfit\FileProcess\SourceData）你自己需要调整设置

    # 输入 \01 和\02 文件夹所在的主目录，n_jobs是多进程的进程数
    file_processor = FileProcess(root_path=r'E:\SIOA\Program\PersonalProfit\FileProcess\SourceData',
                                 n_jobs=4)
    file_processor.run_engine()
    

    