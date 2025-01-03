import cv2
import os
import tqdm
import pickle


# 标签对照表
label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12,
              'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}

# 预处理数据的保存目录
save_path = r"D:\linshi\shixun\archive2\pickle_file"
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 记录x1和x2
x1_lst = []
y_lst = []


# 通过路径处理
def propre_input1_input2_with_path(img_path, flag):
    if img_path.strip() == '':
        return
    # 读入图像
    img = cv2.imread(img_path.split(' ')[0])
    # plt.figure()
    # plt.imshow(img[:, :, ::-1])
    x1 = propre_input1_input2(img)
    # 对处理好的数据进行保存
    # plt.figure()
    # plt.imshow(x1)
    # plt.show()
    x1_lst.append(x1)
    y_lst.append(label_dict[img_path.split(' ')[-1]])
    # print(x1.shape, label_dict[img_path.split(' ')[-1]])


# 预处理input1和input2输入
def propre_input1_input2(img):
    # 改变大小为28x28
    img_gray_28 = cv2.resize(img, (28, 28))
    # 归一化
    img_res = img_gray_28 / 255

    return img_res[:, :, ::-1]


# 进行数据预处理
def propre_data_labels(img_path_and_label, flag):
    # 进行预处理
    for img_path in tqdm.tqdm(img_path_and_label, desc=f"{flag}_图像数据预处理与保存"):
        propre_input1_input2_with_path(img_path, flag)


# 加载路径并预处理
def load_data(data_path, flag):
    with open(data_path) as fp:
        img_path_and_label = fp.read().split('\n')

    propre_data_labels(img_path_and_label, flag)


for name in ["train", "valid", "test"]:
    print(f"D:\\linshi\\shixun\\archive2\\split_text\\{name}_data_random60.txt", "to", os.path.join(save_path, f"sign_language_{name}_random60.pickle"))
    load_data(f"D:\\linshi\\shixun\\archive2\\split_text\\{name}_data_random60.txt", 0)
    # 保存pickle文件
    fp = open(os.path.join(save_path, f"sign_language_{name}_random60.pickle"), 'wb')
    print(len(x1_lst), len(y_lst))
    pickle.dump([x1_lst, y_lst], fp)
    fp.close()
    x1_lst.clear()
    y_lst.clear()