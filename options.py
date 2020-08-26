import datetime
# 데이터
SIGHT = 25  # 모델이 며칠 전의 확진자 수까지 볼 수 있는가? (=x 개수)
Y_N = 5  # 모델이 한 번에 며칠 후의 확진자 수까지 예측해야 하는가? (=y 개수)

TEST_SIZE = 0.1  # 테스트 데이터 비율

# 학습
EPOCHS = 20
BATCH_SIZE = 256
lr = 0.001

# 예측
POS = -1  # 몇 번째 데이터에서부터 forecast 예측을 할 것인가? (-1이면 미래의 확진자 수만 예측)
DAYS = 300  # 며칠 후의 확진자 수까지 예측할 것인가?
START_DATE = datetime.datetime(2020, 2, 9)  # 대한민국 첫 데이터에 해당하는 날짜
