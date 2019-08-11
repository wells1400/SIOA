import numpy as np
import pandas as pd
import os


# 输入数据目录获取文件路径和文件名
def get_path(data_root=r'E:\SIOA\Program\PersonalProfit\ExcelToTxt\SourceData'):
    file_path = []
    file_name = []
    relative_path = os.listdir(data_root)
    for path in relative_path:
        file_path.append(data_root + '\\' + path)
        file_name.append(path.split('.')[0])
    return file_path,file_name


# 处理一个文件（输入文件路径，文件名，output主目录），输出建立一个文件夹，对文件进行处理
def file_process(file_path, file_name, output_root):
    # 如果该输出目录下以file_name命名的目录存在，则清空里面的文件，
    # 否则直接创建一个新的文件夹
    if os.path.exists(output_root + '\\' + file_name):
        tmp_list = [output_root + '\\' + file_name + '\\' + file for file in os.listdir(output_root + '\\' + file_name)]
        for tmp_file in tmp_list:
            os.remove(tmp_file)
    else:
        os.mkdir(output_root + '\\' + file_name)
    data_pd = pd.read_excel(file_path)
    data_pd.drop(data_pd.loc[data_pd['条件'].isnull()].index,axis=0,inplace=True)
    unique_counts_list = np.unique(data_pd['个数'])
    for index in range(len(unique_counts_list)):
        selected_pd = data_pd.loc[data_pd['个数'] == unique_counts_list[index]]
        data_list = list(selected_pd['条件'].astype(int))
        with open(output_root + '\\' + file_name + '\\' + str(unique_counts_list[index]) + '.txt','w') as f:
            for row in data_list:
                f.write(str(row))
                f.write('\n')


# 运行主程,输入数据目录，输出主目录
def run_engine(data_root=r'E:\SIOA\Program\PersonalProfit\ExcelToTxt\SourceData',
               output_root=r'E:\SIOA\Program\PersonalProfit\ExcelToTxt\OutputData'
               ):
    file_path_list,file_name_list = get_path(data_root=data_root)
    for index in range(len(file_path_list)):
        file_path = file_path_list[index]
        file_name = file_name_list[index]
        print(r'>>>处理%s>>>'%file_name)
        file_process(file_path, file_name, output_root)
    print(r'>>>处理完成！>>>')


if __name__ == '__main__':
    # data_root是存放待处理的Excel文件的目录，output_root是存放输出结果的主目录
    run_engine(data_root=r'E:\SIOA\Program\PersonalProfit\ExcelToTxt\SourceData',
               output_root=r'E:\SIOA\Program\PersonalProfit\ExcelToTxt\OutputData')