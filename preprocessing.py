# 데이터를 가져온다.
# 학습 직전의 데이터를 가공하는 역할을 맡는다.
import numpy as np
import os
import options as opt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import joblib


def date_to_number(date):
    # 글자로 표현된 날짜를 숫자로 바꾼다.
    # ex) 'Dec/1' -> '12/1'
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months):
        date = date.replace(month, str(i+1))
    return date


def load_one(path):
    # path의 txt 파일 데이터를 가져온다.
    with open(path, 'r', encoding='utf8') as f:
        d1 = f.read().split(' ')

    d1 = d1[d1.count('nan'):]
    d1 = list(map(float, d1))
    return d1


def load(directory='data'):
    # directory의 txt 파일 데이터들을 가져온다.
    data = []
    for file in os.listdir(directory):
        path = directory + '/' + file
        with open(path, 'r', encoding='utf8') as f:
            d1, d2 = [i.split(' ') for i in f.read().split('\n')]

        d1 = d1[d1.count('nan'):]  # nan 제거
        d2 = d2[d2.count('nan'):]
        d1 = list(map(float, d1))
        d2 = list(map(float, d2))
        data += d1
        data += d2

    return np.array(data)


def labeling(data, sight=25, y_n=1):
    # (sight개의 x, y_n개의 y) 쌍을 만든다.
    # ex) f([1,2,3,4,5,6,7,8,9], sight=3, y_n=1) -> [ [[1,2,3],[4]], [[2,3,4],[5]], ..]
    x, y = [], []
    for i in range(len(data) - sight - y_n + 1):
        x.append(data[i:sight+i])
        y.append(data[sight+i:sight+i+y_n])
    return np.array(x), np.array(y)


if __name__ == "__main__":
    # 데이터 가져오기
    data = load()
    # x, y 제작
    x, y = labeling(data, sight=opt.SIGHT, y_n=opt.Y_N)
    x = x.reshape((-1, opt.SIGHT, 1))

    # train, test 데이터로 분할
    xtrain, xtest, ytrain ,ytest = train_test_split(x, y, test_size=opt.TEST_SIZE)
    joblib.dump([xtrain, xtest, ytrain, ytest], 'traintest.joblib')  # 저장

    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)
