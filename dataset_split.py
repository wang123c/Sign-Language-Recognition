import os
from sklearn.model_selection import train_test_split


"""
对数据集进行整理分割，将训练数据和验证数据和测试数据的路径，分别放到不同的txt文件下
"""
# 根路径
root_path = r"D:\linshi\shixun\archive1\dataset5"

# 记录列表
data_dict = dict()

# 分割比例
test_size = 0.2
valid_size = 0.25

# 保存路径
dir_path = r"D:\linshi\shixun\archive2\split_text"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


# 找出全部的数据并分类
def split_dataset(path):
    temp_lst_path = os.listdir(path)
    if len(temp_lst_path) == 0: return
    # 判断是否还是目录
    if os.path.isdir(os.path.join(path, temp_lst_path[0])):
        for temp_path in temp_lst_path:
            split_dataset(os.path.join(path, temp_path))
    else:
        label = os.path.basename(path)
        if label not in data_dict.keys():
            data_dict[label] = []
        for temp_path in temp_lst_path:
            if temp_path.split("_")[0] == "color":
                data_dict[label].append(os.path.join(path, temp_path))


# 对数据进行分割并保存路径
def split_save():
    for k, v in data_dict.items():
        tol = len(v)
        train_path, test_path = train_test_split(v, test_size=test_size, random_state=60)
        train_path, valid_path = train_test_split(train_path, test_size=valid_size, random_state=60)
        with open(os.path.join(dir_path, "train_data_random60.txt"), 'a', encoding='utf-8') as fp:
            for i in train_path:
                fp.write(f"{i} {k}\n")
        with open(os.path.join(dir_path, "valid_data_random60.txt"), 'a', encoding='utf-8') as fp:
            for i in valid_path:
                fp.write(f"{i} {k}\n")
        with open(os.path.join(dir_path, "test_data_random60.txt"), 'a', encoding='utf-8') as fp:
            for i in test_path:
                fp.write(f"{i} {k}\n")


if __name__ == '__main__':
    split_dataset(root_path)
    split_save()

