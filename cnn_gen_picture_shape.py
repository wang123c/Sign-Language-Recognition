## ============载入数据================
import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Input, Add, AveragePooling2D, MaxPooling2D, Reshape


num_classes = 24
batch_size = 64
epoch = 100

label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
              'g': 6,'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11,
              'n': 12,'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17,
              't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}

# one_hot处理
def data_one_hot(x, y):
    y = tf.one_hot(y, depth=len(label_dict))
    return x, y

# =======加载.pickle文件数据集=======
def load_data_pickle(data_path):
    with open(os.path.join(data_path, "sign_language_train_random60.pickle"), 'rb') as fp:
        train_input, train_label = pickle.load(fp)
        train_input = [tf.constant(i) for i in train_input]
        train_label = [tf.constant(i) for i in train_label]
    with open(os.path.join(data_path, "sign_language_valid_random60.pickle"), 'rb') as fp:
        valid_input, valid_label = pickle.load(fp)
        valid_input = [tf.constant(i) for i in valid_input]
        valid_label = [tf.constant(i) for i in valid_label]
    with open(os.path.join(data_path, "sign_language_test_random60.pickle"), 'rb') as fp:
        test_input1, test_label = pickle.load(fp)
        test_input = [tf.constant(i) for i in test_input1]
        test_label = [tf.constant(i) for i in test_label]
    print("pickle数据加载完毕,开始转为Dataset对象")
    print(f"可用数据量：trian:{len(train_input)}, valid:{len(valid_input)}, test:{len(test_input)}")
    # 将数据处理称Dataset对象
    time1 = time.time()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label)).map(data_one_hot) \
        .shuffle(10000).batch(batch_size)
    print(f"训练数据集train_dataset准备完成, 用时：{time.time() - time1:.2f}s")
    time1 = time.time()
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_label)).map(data_one_hot) \
        .batch(batch_size)
    print(f"测试数据集valid_dataset准备完成, 用时{time.time() - time1:.2f}s")
    time1 = time.time()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_label)).map(data_one_hot) \
        .batch(batch_size)
    print(f"测试数据集test_dataset准备完成, 用时{time.time() - time1:.2f}s")
    return train_dataset, valid_dataset, test_dataset, test_label, test_input1


# 卷积相乘
def multiply_layer(x):
    xx = x[0] * x[1]
    return xx

def subtract_layer_caculation(x):
    xx = x[0] * x[1]/2 + x[0] * x[1]/2
    return xx

# =======创建权重相加卷积模型=======
def cnn_model_weight():
    input_img = Input(shape=(28, 28, 3))
    con1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    con1 = BatchNormalization()(con1)
    con1 = MaxPool2D((2, 2), strides=2, padding="same")(con1)

    con2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(con1)
    con2 = Dropout(0.2)(con2)
    con2 = BatchNormalization()(con2)
    con2 = MaxPool2D((2, 2), strides=2, padding="same")(con2)

    con3 = Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(con2)
    con3 = Dropout(0.2)(con3)
    con3 = BatchNormalization()(con3)
    con3 = MaxPool2D((2, 2), strides=2, padding="same")(con3)

    con4 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con4', con4)
    con4_flat = Flatten()(con4)
    # print('con4_flat', con4_flat)
    den1 = Dense(units=1024, activation="relu")(con4_flat)
    den1 = Dense(units=2048, activation="relu")(den1)
    den1 = BatchNormalization()(den1)
    act1 = Activation('sigmoid')(den1)
    # print('act1', act1)
    act1 = Reshape((4, 4, 128))(act1)
    # print('re_act1', re_act1)
    # subtract_layer1 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])
    # subtracted_output1 = subtract_layer1([con4, act1])
    subtracted_output1 = multiply_layer([con4, act1])

    con5 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con5', con5)
    con5_flat = Flatten()(con5)
    # print('con5_flat', con5_flat)
    den2 = Dense(units=1024, activation="relu")(con5_flat)
    den2 = Dense(units=2048, activation="relu")(den2)
    den2 = BatchNormalization()(den2)
    act2 = Activation('sigmoid')(den2)
    # print('act2', act2)
    act2 = Reshape((4, 4, 128))(act2)
    # print('re_act1', re_act1)
    # subtract_layer2 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])
    # subtracted_output2 = subtract_layer2([con5, act2])
    subtracted_output2 = multiply_layer([con5, act2])

    add1 = Add()([subtracted_output1, subtracted_output2])
    add1 = BatchNormalization()(add1)
    add1 = AveragePooling2D(2)(add1)
    add1 = BatchNormalization()(add1)
    add1 = MaxPooling2D(2)(add1)

    con6 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(add1)
    con6 = BatchNormalization()(con6)
    con6 = MaxPool2D((2, 2), strides=2, padding="same")(con6)


    flat = Flatten()(con6)
    dens1 = Dense(units=256, activation="relu")(flat)
    dens1 = Dropout(0.3)(dens1)

    dens2 = Dense(units=128, activation="relu")(dens1)
    dens2 = Dropout(0.3)(dens2)

    dens3 = Dense(units=num_classes, activation="softmax")(dens2)

    model = Model(inputs=input_img, outputs=dens3)


    ## ==========模型编译==========
    sgd = SGD(learning_rate=0.01)
    rms = RMSprop(learning_rate=0.001)
    ada = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    model.summary()
    return model

# =======创建相加卷积模型=======
def cnn_model_add():
    input_img = Input(shape=(28, 28, 3))
    con1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    con1 = BatchNormalization()(con1)
    con1 = MaxPool2D((2, 2), strides=2, padding="same")(con1)

    con2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(con1)
    con2 = Dropout(0.2)(con2)
    con2 = BatchNormalization()(con2)
    con2 = MaxPool2D((2, 2), strides=2, padding="same")(con2)

    con3 = Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(con2)
    con3 = Dropout(0.2)(con3)
    con3 = BatchNormalization()(con3)
    con3 = MaxPool2D((2, 2), strides=2, padding="same")(con3)

    con4 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con4', con4)

    con5 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con5', con5)

    add1 = Add()([con4, con5])
    add1 = BatchNormalization()(add1)
    add1 = AveragePooling2D(2)(add1)
    add1 = BatchNormalization()(add1)
    add1 = MaxPooling2D(2)(add1)

    con6 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(add1)
    con6 = BatchNormalization()(con6)
    con6 = MaxPool2D((2, 2), strides=2, padding="same")(con6)

    flat = Flatten()(con6)
    dens1 = Dense(units=256, activation="relu")(flat)
    dens1 = Dropout(0.3)(dens1)

    dens2 = Dense(units=128, activation="relu")(dens1)
    dens2 = Dropout(0.3)(dens2)

    dens3 = Dense(units=num_classes, activation="softmax")(dens2)

    model = Model(inputs=input_img, outputs=dens3)

    ## ==========模型编译==========
    sgd = SGD(learning_rate=0.01)
    rms = RMSprop(learning_rate=0.001)
    ada = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    model.summary()
    return model

# =======创建卷积模型=======
def cnn_model():
    input_img = Input(shape=(28, 28, 3))
    con1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    con1 = BatchNormalization()(con1)
    con1 = MaxPool2D((2, 2), strides=2, padding="same")(con1)

    con2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(con1)
    con2 = Dropout(0.2)(con2)
    con2 = BatchNormalization()(con2)
    con2 = MaxPool2D((2, 2), strides=2, padding="same")(con2)

    con3 = Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(con2)
    con3 = Dropout(0.2)(con3)
    con3 = BatchNormalization()(con3)
    con3 = MaxPool2D((2, 2), strides=2, padding="same")(con3)

    con6 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    con6 = BatchNormalization()(con6)
    con6 = MaxPool2D((2, 2), strides=2, padding="same")(con6)

    flat = Flatten()(con6)
    dens1 = Dense(units=256, activation="relu")(flat)
    dens1 = Dropout(0.3)(dens1)

    dens2 = Dense(units=128, activation="relu")(dens1)
    dens2 = Dropout(0.3)(dens2)

    dens3 = Dense(units=num_classes, activation="softmax")(dens2)

    model = Model(inputs=input_img, outputs=dens3)

    ## ==========模型编译==========
    sgd = SGD(learning_rate=0.01)
    rms = RMSprop(learning_rate=0.001)
    ada = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    model.summary()
    return model


def cnn_model_caculation():
    input_img = Input(shape=(28, 28, 3))
    con1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    con1 = BatchNormalization()(con1)
    con1 = MaxPool2D((2, 2), strides=2, padding="same")(con1)

    con2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(con1)
    con2 = Dropout(0.2)(con2)
    con2 = BatchNormalization()(con2)
    con2 = MaxPool2D((2, 2), strides=2, padding="same")(con2)

    con3 = Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(con2)
    con3 = Dropout(0.2)(con3)
    con3 = BatchNormalization()(con3)
    con3 = MaxPool2D((2, 2), strides=2, padding="same")(con3)

    con4 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con4', con4)

    con5 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(con3)
    # print('con5', con5)

    add1 = subtract_layer_caculation([con4, con5])
    con6 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(add1)
    con6 = BatchNormalization()(con6)
    con6 = MaxPool2D((2, 2), strides=2, padding="same")(con6)

    flat = Flatten()(con6)
    dens1 = Dense(units=256, activation="relu")(flat)
    dens1 = Dropout(0.3)(dens1)

    dens2 = Dense(units=128, activation="relu")(dens1)
    dens2 = Dropout(0.3)(dens2)

    dens3 = Dense(units=num_classes, activation="softmax")(dens2)

    model = Model(inputs=input_img, outputs=dens3)

    ## ==========模型编译==========
    sgd = SGD(learning_rate=0.01)
    rms = RMSprop(learning_rate=0.001)
    ada = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    model.summary()
    return model


# ==========loss曲线==========
def loss_accuracy(history, dir_path, cnt, model_name):
   # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], color='blue', linestyle='-', marker='o', markerfacecolor='none', label='Training Loss')
    # plt.plot(history.history['val_loss'], color='red', linestyle='-', marker='o', markerfacecolor='none', label='Validation Loss')

    plt.plot(history.history['loss'], color='blue', linestyle='-', marker='o', label='Training Loss')
    plt.plot(history.history['val_loss'], color='red', linestyle='-', marker='o', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], color='blue', linestyle='-', marker='o', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='red', linestyle='-', marker='o', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, str(cnt), f"{model_name}_training_curves.png"))
    plt.show()

# ==========混淆矩阵==========
def confusion(y_test, y_pred, dir_path, cnt, model_name):
    labels = ['A', 'B', 'C', 'D', 'E', 'F',
              'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'X', 'Y']

    # 计算混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)

    # 使用seaborn的heatmap函数绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # 设置标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(dir_path, str(cnt), f"{model_name}_training_confusion.png"))
    # 显示图形
    plt.show()



def canshu(y_test, y_pred_class, dir_path, cnt, model_name):
    y_test = np.array(y_test)
    y_pred_class = np.array(y_pred_class)

    # 计算每个类别的精确率、召回率和F1-Score
    precisions = precision_score(y_test, y_pred_class, average=None, labels=np.arange(num_classes), zero_division=0)
    recalls = recall_score(y_test, y_pred_class, average=None, labels=np.arange(num_classes), zero_division=0)
    f1_scores = f1_score(y_test, y_pred_class, average=None, labels=np.arange(num_classes), zero_division=0)

    # 计算Micro Average，Macro Average，Weighted Average
    precision_micro = precision_score(y_test, y_pred_class, average='micro', zero_division=0)
    recall_micro = recall_score(y_test, y_pred_class, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred_class, average='micro', zero_division=0)

    precision_macro = precision_score(y_test, y_pred_class, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred_class, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred_class, average='macro', zero_division=0)

    precision_weighted = precision_score(y_test, y_pred_class, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred_class, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_class, average='weighted', zero_division=0)

    # 创建一个DataFrame来保存结果
    data = {
        'Label': label_dict.values(),
        'Class': np.arange(num_classes),
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
        'Number of Actual Predictions': [np.sum(y_test == i) for i in np.arange(num_classes)]
    }
    df = pd.DataFrame(data)

    # 添加Micro Average，Macro Average，Weighted Average行
    micro_data = {
        'Label': 'Micro Average',
        'Class': None,
        'Precision': precision_micro,
        'Recall': recall_micro,
        'F1-Score': f1_micro,
        'Number of Actual Predictions': None
    }
    macro_data = {
        'Label': 'Macro Average',
        'Class': None,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1-Score': f1_macro,
        'Number of Actual Predictions': None
    }
    weighted_data = {
        'Label': 'Weighted Average',
        'Class': None,
        'Precision': precision_weighted,
        'Recall': recall_weighted,
        'F1-Score': f1_weighted,
        'Number of Actual Predictions': None
    }
    df = pd.concat([df, pd.DataFrame([micro_data]), pd.DataFrame([macro_data]), pd.DataFrame([weighted_data])], ignore_index=True)

    # 保存DataFrame到CSV文件
    df.to_csv(os.path.join(dir_path, str(cnt), f"{model_name}_classification_metrics.csv"), index=False)

    # 打印DataFrame
    print('classification_metrics.csv 保存完成')



# ==========模型训练==========
def model_train(model_creation, model_name):
    # 设置相同的种子值可以使得每次运行代码时，涉及到随机性的部分（如权重初始化、数据集的随机划分等）产生相同的结果，这有助于实验的可重复性。
    # tf.random.set_seed(10)
    # tf.config.experimental.enable_op_determinism()
    data_path = r"E:\linshi\shixun\archive2\pickle_file"
    train_dataset, valid_dataset, test_dataset, test_label, test_input1 = load_data_pickle(data_path)
    model = model_creation()

    dir_path = os.path.join("./train_results",
                            f"train_result_sign_{model_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    cnt = 0
    for i in os.listdir(dir_path):
        cnt = max(cnt, int(i))
    cnt += 1
    os.mkdir(os.path.join(dir_path, str(cnt)))
    TensorBoard_dir = os.path.join(dir_path, str(cnt), f"TensorBoard")
    ModelCheckpoint_dir = os.path.join(dir_path, str(cnt), f"ModelCheckpoint.keras")

    callback = [
        callbacks.TensorBoard(TensorBoard_dir),
        callbacks.ModelCheckpoint(filepath=ModelCheckpoint_dir, save_best_only=True, monitor="val_accuracy",
                                  mode='max'),
        # callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5),
        ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
    ]
    time1 = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}开始训练")
    history = model.fit(train_dataset, epochs=epoch, verbose=1,
                        validation_data=valid_dataset, callbacks=callback, shuffle=False)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}训练完成！用时：{time.time() - time1}")

    # 评估
    model = keras.models.load_model(ModelCheckpoint_dir)
    score = model.evaluate(test_dataset, batch_size=20)
    print('loss:', score[0])
    print('准确率:', score[1])

    # 模型保存
    model.save_weights(
        os.path.join(dir_path, str(cnt), f"model_weights_{time.strftime('%Y_%m_%d_%H_%M_%S')}.weights.h5"))
    print("模型保存完成")
    with open(os.path.join("测试结果.txt"), 'a') as fp:
        fp.write(f"{model_name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}" + str(score) + "\n")
    print("测试结果.txt保存完成")
    loss_accuracy(history, dir_path, cnt, model_name)
    # 预测
    result = model.predict(test_dataset)
    y_pred = tf.argmax(result, axis=1)
    confusion(test_label, y_pred, dir_path, cnt, model_name)
    canshu(test_label, y_pred,  dir_path, cnt, model_name)

# ==========模型训练==========
if __name__ == "__main__":
    model_train(cnn_model_weight, "cnn_model_weight")
    # model_train(cnn_model_add, "cnn_model_add")
    # model_train(cnn_model, "cnn_model")
    # model_train(cnn_model_caculation, "cnn_model_caculation")
