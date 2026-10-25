import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "inputdata.csv"
MODEL_PATH = BASE_DIR / "C_L_A.h5"


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print('shape为', layer_activations.shape)
            print(layer_activations)
    return activations


# 提取模型在给定输入inputs下，各层的输出值，用于分析模型内部的特征提取过程

def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


# 生成简单的结构化数据，用于测试注意力机制是否能自动识别 "与目标相关的关键特征列"

def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network    should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


# 生成时序数据（适合 LSTM、注意力时序模型等），测试注意力机制是否能聚焦 "与目标相关的关键时间步"

def get_data_recurrent2(n, time_steps, input_dim, attention_dim=5):
    """
    Suppose input_dim = 10  time_steps = 6
    formed one  x 6 x 10 The data of each step 6 dimension is the same as y
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, :, attention_dim] = np.tile(y[:], (1, time_steps))
    # 生成另一种时序数据，测试注意力机制是否能聚焦 "与目标相关的关键特征维度"（而非时间步）

    return x, y


from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Model

# from attention_utils import get_activations
from keras.layers import Multiply
from keras.layers.core import *
from keras.models import *

import pandas as pd
import numpy as np

from keras.layers.merging.concatenate import concatenate
from keras.layers.merging.add import add
from keras.layers.merging.multiply import multiply
from keras.layers.merging.subtract import subtract
from keras.layers.merging.average import average
from keras.layers.merging.maximum import maximum
from keras.layers.merging.minimum import minimum
from keras.layers.merging.dot import dot

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)输入是形状为(batch_size, time_steps, input_dim)的张量（通常是 LSTM 的输出）
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def attention_3d_block2(inputs, single_attention_vector=False):
    # If the upper layer is LSTM, you need return_sequences=True另一种注意力实现方式，先通过Permute交换特征维度和时间维度
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # Multiplied by the attention weight, but there is no summation, it seems to have little effect
    # If you classify tasks, you can do Flatten expansion
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# 将时间序列数据转换为监督学习格式
def create_dataset(dataset, look_back):
    '''
    Processing the data
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


# Multidimensional normalization returns data and maximum and minimum values归一化
def NormalizeMult(data):
    # normalize Used for denormalization
    data = np.array(data)
    normalize = np.arange(2 * data.shape[1], dtype='float64')

    normalize = normalize.reshape(data.shape[1], 2)
    print(normalize.shape)
    for i in range(0, data.shape[1]):
        # Column i
        list = data[:, i]
        listlow, listhigh = np.percentile(list, [0, 100])
        # print(i)
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            # Row j
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    # np.save("./normalize.npy",normalize)
    return data, normalize


# Multidimensional denormalization反归一化
def FNormalizeMult(data, normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            # Row j
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


# 核心模型定义，结合了 CNN、双向 LSTM 和注意力机制
def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(drop)(x)
    # filters控制提取的局部特征数量，需结合数据复杂度，特征多则增大，反之减小，该值最好是2的n次方。kernel_size是连续时间步长度，可尝试增加尝试，保证≤TIME_STEPS即可
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # For GPU you can use CuDNNLSTM
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(drop)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    # output = Dense(1, activation='sigmoid')(attention_mul)
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

data = pd.read_csv(DATA_PATH)
data

data = data.drop(['time'], axis = 1)
print(data.columns)
print(data.shape)

INPUT_DIMS = 9#输入数据的特征维度（即每个时间步包含的特征，这里是除了前两列的列数）
TIME_STEPS = 20#时间步长（使用连续多少个时间步的特征数据来预测目标值，即用来预测的数据行数），增大可以增加精度，但要注意测试集数量比他大
lstm_units = 64#LSTM 层的隐藏单元数量（决定模型的学习能力）可以尝试 32、128、256（需平衡，过大会增加训练难度）
drop = 0#Dropout 层的丢弃比例（这里设为 0，表示不使用 Dropout 防止过拟合）可以尝试调整为0.1或0.2等等

#Normalized
data,normalize = NormalizeMult(data)
Deformed_data = data[:,0].reshape(len(data),1)

train_X, _ = create_dataset(data,TIME_STEPS)
_ , train_Y = create_dataset(Deformed_data,TIME_STEPS)

print(train_X.shape,train_Y.shape)

m = attention_model()
m.summary()
m.compile(optimizer='adam', loss='mse')
#m.fit([train_X], train_Y, epochs=10, batch_size=64, validation_split=0.1)
#m.save("./model.h5")
#np.save("normalize.npy",normalize)

from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)
df = df.drop(['time'], axis=1)
# 确保测试集包含最后一行，且长度足够生成包含最后一行的样本（需至少TIME_STEPS + 1行）
train_size = int(len(df) * 0.8)  # 数据划分
train = df.iloc[:train_size, :]
test = df.iloc[train_size:, :]
# train, test = train_test_split(df, test_size=0.1)
print("len(train):", len(train))
print("len(test):", len(test))

import keras.backend as K


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_path = str(BASE_DIR / 'model.h5')
# TRAIN
INPUT_DIMS = 9#如果修改输入列数，这里要修改为除前两列的其他列数
TIME_STEPS = 20
lstm_units = 64

#Normalized
train,normalize = NormalizeMult(train)
Deformed_data = train[:,0].reshape(len(train),1)

train_X, _ = create_dataset(train,TIME_STEPS)
_ , train_Y = create_dataset(Deformed_data,TIME_STEPS)
print(train_X.shape,train_Y.shape)

m = attention_model()
m.summary()
#m.compile(optimizer='adam', loss='mse')
m.compile(loss='mae', optimizer='adam',metrics=['mae',r2_keras])

# fit the network
history =  m.fit([train_X], train_Y, epochs=30, batch_size=10, validation_split=0.1)
#history = m.fit([train_X], train_Y, epochs=10, batch_size=64, validation_split=0.05, verbose=2,
#          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
#                       ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
#          )
#epochs可以适当增加或减少，batch_size可以适当增大，注意设置为2的N次方

m.save(str(MODEL_PATH))
import matplotlib.pyplot as plt

test_np = test.values  # test是之前划分的测试集（df.iloc[train_size:, :]）
test_normalized = test_np.copy()  # 复制测试集用于归一化处理

# 用训练集的归一化参数（normalize）对测试集每一列进行归一化
for i in range(test_normalized.shape[1]):
    # 训练集第i列的最小值和最大值（来自train的归一化结果）
    listlow = normalize[i, 0]
    listhigh = normalize[i, 1]
    delta = listhigh - listlow
    if delta != 0:  # 避免除以0
        test_normalized[:, i] = (test_normalized[:, i] - listlow) / delta

# 2. 从归一化后的test数据中提取目标列（假设第0列是需要预测的目标，与训练集一致）
Deformed_test = test_normalized[:, 0].reshape(len(test_normalized), 1)  # 测试集目标列（归一化后）

# 3. 用test数据生成test_X（输入）和test_Y（真实标签）
# create_dataset函数：用前TIME_STEPS行数据预测第TIME_STEPS+1行的目标值
test_X, _ = create_dataset(test_normalized, TIME_STEPS)  # 测试输入（来自test数据）
_, test_Y = create_dataset(Deformed_test, TIME_STEPS)     # 测试目标（来自test数据的目标列）

# 检查测试样本数量是否足够（需至少生成1个样本）
if len(test_X) == 0:
    print(f"测试集样本不足！需要至少{TIME_STEPS + 1}行数据，当前test数据有{len(test)}行")
else:
    print(f"测试集样本形状：test_X={test_X.shape}, test_Y={test_Y.shape}")

    # 4. 用模型预测测试集
    scores_test = m.evaluate([test_X], test_Y, verbose=2)
    results = m.predict([test_X])

# Prediction
scores_test = m.evaluate([test_X], test_Y, verbose=2)

results = m.predict([test_X])

test_new=[]
predict_new=[]

for k in range(len(results)):
    if test_Y[k]!=0:
        test_new.append(test_Y[k])
        predict_new.append(results[k])

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.fabs((y_true - y_pred) / y_true))
mape = format(MAPE(test_new, predict_new), '.4f')
mape

# 下一步预测
import keras.backend as K  # 确保Keras后端已导入（如果已有可忽略）


# 必须重新定义或导入r2_keras函数（确保在加载模型前可见）
def r2_keras(y_true, y_pred):
    """Coefficient of Determination"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# 加载模型时指定自定义指标
model = load_model(str(MODEL_PATH), custom_objects={'r2_keras': r2_keras})


def predict_next_row(model, data, time_steps, normalize):  # 注意参数改为 normalize
    """
    预测下一行数据
    model: 训练好的模型
    data: 原始数据（用于预测的历史数据）
    time_steps: 时间步长
    normalize: 训练集的归一化参数（来自NormalizeMult的返回值）
    """
    # 1. 取最后time_steps行数据作为输入（与训练时的输入格式一致）
    last_sequence = data[-time_steps:]  # 形状：(time_steps, 特征数)

    # 2. 用训练集的归一化参数对输入数据进行归一化（和测试集处理方式一致）
    last_sequence_scaled = last_sequence.copy()
    for i in range(last_sequence_scaled.shape[1]):
        listlow = normalize[i, 0]  # 训练集第i列的最小值
        listhigh = normalize[i, 1]  # 训练集第i列的最大值
        delta = listhigh - listlow
        if delta != 0:
            last_sequence_scaled[:, i] = (last_sequence_scaled[:, i] - listlow) / delta

    # 3. 调整输入形状为模型所需格式 [samples, time_steps, features]
    X_input = last_sequence_scaled.reshape(1, time_steps, -1)  # 增加样本维度

    # 4. 预测下一行数据（归一化尺度）
    next_row_scaled = model.predict(X_input)  # 形状：(1, 1)

    # 5. 反归一化，恢复原始数据尺度（使用FNormalizeMult函数）
    next_row_2d = next_row_scaled.reshape(-1, 1)  # 转换为二维数组（符合FNormalizeMult的输入要求）
    next_row = FNormalizeMult(next_row_2d, normalize)  # 反归一化

    return next_row


if __name__ == "__main__":
    # 确保以下变量已在代码中定义（从训练过程中获取）
    # model（已加载的模型）、df（原始数据）、TIME_STEPS=20、normalize（训练集的归一化参数）

    # 加载模型（之前修改过的加载代码，确保包含custom_objects）
    def r2_keras(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    model = load_model(str(MODEL_PATH), custom_objects={'r2_keras': r2_keras})  # 关键：加载模型时传入自定义指标

    # 原始数据（假设df是你的全部数据）
    data = df.values  # 将DataFrame转换为numpy数组

    # 预测下一行数据（传入normalize参数，而非scaler）
    predicted_row_201 = predict_next_row(model, data, TIME_STEPS, normalize)  # 这里用normalize

    # 输出预测结果
    print("第201行的预测数据：")
    print(predicted_row_201)

    # 保存到CSV文件
    import pandas as pd

    pd.DataFrame(predicted_row_201).to_csv(BASE_DIR / "predicted_row_201.csv", index=False, header=False)

