import time
import os


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
    print('End at:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with open(file_path_3, 'w') as f:
        for val in res_data:
            f.write(val)
            f.write('\n')


# 运行主程序
def run_engine(root_path=r'E:\SIOA\Program\PersonalProfit\FileProcess\SourceData'):
    test_count = int(input(r'请输入要循环测次数：'))
    file_path_1_index = input(r'请输入file_path_1名：')
    file_path_2_index = input(r'请输入file_path_2名：')

    file_path_1 = root_path + r'\\' + file_path_1_index + r'\\'
    file_path_2 = root_path + r'\\' + file_path_2_index + r'\\'

    out_file_index = 1
    iter_count = 0
    # 读取file_path_1文件夹内所有txt文件
    files_list = os.listdir(file_path_1)
    files_entire_list = [file_path_1 + files_list[i] for i in range(len(files_list))]
    while True:
        if iter_count >= test_count:
            break
        # 要计算的两个文件的索引
        file_index_1 = iter_count // (len(files_list))
        file_index_2 = iter_count % (len(files_list))
        # 要计算的两个文件的名称
        base_name_1 = files_list[file_index_1].split(r'.')[0]
        base_name_2 = files_list[file_index_2].split(r'.')[0]
        # 输出文件
        file_path_3 = file_path_2 + str(out_file_index) + r'_' + base_name_1 + '_to_' + base_name_2 + r'.txt'
        print(r'file_path_1: <<  %s >>' % files_entire_list[file_index_1])
        print(r'file_path_2: <<  %s >>' % files_entire_list[file_index_2])
        print(r'file_path_3: <<  %s >>' % file_path_3)
        file_process(files_entire_list[file_index_1], files_entire_list[file_index_2], file_path_3)
        out_file_index += 1
        iter_count += 1


if __name__ == '__main__':
    # 备注：：：：：
    # 要手动输入的是要计算的文件所放的文件夹（file_path_1）,以及输出文件需要存放的文件夹（file_path_2）这两个文件夹的名字
    # 以这里的测试数据为例，file_path_1输入是01， file_path_2输入时02
    # 然后root_path是主目录（即01和02文件夹的父文件夹的绝对路径，
    # 我这里是E:\SIOA\Program\PersonalProfit\FileProcess\SourceData）你自己需要调整设置88

    run_engine(root_path=r'E:\SIOA\Program\PersonalProfit\FileProcess\SourceData')
    

    