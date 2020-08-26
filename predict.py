import tensorflow as tf
import os
from models import build_model
import preprocessing as pre
import options as opt
import numpy as np
import matplotlib.pyplot as plt
import datetime


# warning 무시
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def forecast(x, days=10):
    # x값을 연속적으로 days번 예측한다.
    result = []
    x = x.reshape(-1, 1)
    for _ in range(days):
        pred = model.predict(x.reshape((1, -1, 1)))[0][0]
        result.append(pred)
        x = np.append(x[1:], pred)
    return result


def load_korea_data():
    # 대한민국 데이터를 가져온다.
    data = pre.load_one('south-korea_data.txt')
    x, y = pre.labeling(data, sight=opt.SIGHT, y_n=opt.Y_N)
    x = x.reshape((-1, opt.SIGHT, 1))
    full_x = np.array([data[i:opt.SIGHT + i] for i in range(len(data) - opt.SIGHT + 1)]).reshape((-1, opt.SIGHT, 1))
    full_y = y[:, 0]
    if opt.Y_N > 1:
        full_y = np.append(x[0], full_y)
        full_y = np.append(full_y, y[-1])
    return x, y, full_x, full_y


def get_dates(last_x):
    # x에 해당하는 40일  날짜를 반환한다.
    dates = []
    for day in range(0, last_x, 40):
        date = opt.START_DATE + datetime.timedelta(day)
        date = date.ctime().replace('  ', ' ').split(' ')[1:3]
        date = pre.date_to_number(date[0]) + '/' + date[1]
        dates.append(date)
    return dates


if __name__ == "__main__":
    # 모델 불러오기
    model = build_model()
    model.load_weights('weights.h5')

    # 대한민국 데이터 가져오기
    x, y, full_x, full_y = load_korea_data()

    # 예측
    preds = model.predict(full_x)  # 'predict' 예측

    if opt.POS >= 0:  # POS는 반드시 음수여야 한다
        opt.POS -= len(x)
    forecast_x = np.append(x[opt.POS][-opt.SIGHT+opt.Y_N:], y[opt.POS][:])
    forecast_result = forecast(forecast_x, days=opt.DAYS)  # 'forecast' 예측

    # 시각화
    plt.plot(full_y, color='b')  # 'real data'
    plt.plot(np.arange(opt.SIGHT, opt.SIGHT+len(preds)), [i[0] for i in preds.reshape(-1, opt.Y_N)], 'r:')  # 'predict'

    last_x = len(x)+opt.SIGHT+opt.Y_N+1+opt.POS+opt.DAYS
    dates = get_dates(last_x)
    plt.plot(range(len(x)+opt.SIGHT+opt.Y_N+opt.POS, last_x),
             np.append([y[opt.POS][-1]], forecast_result), color='g', alpha=0.7)  # 'forecast'

    plt.title('As of 2020-08-26')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.xticks(range(0, last_x, 40), dates)  # x축 Date
    plt.legend(['real data', 'predict', 'forecast'], prop={'size': 12})
    plt.grid(True, alpha=0.5, linestyle='-', linewidth=1)
    plt.show()
