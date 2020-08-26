import tensorflow as tf
import os
from models import build_model
import joblib
import options as opt

# warning 무시
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# 데이터 가져오기
xtrain, xtest, ytrain, ytest = joblib.load('traintest.joblib')

# 모델 가져오기
model = build_model()
try:
    # 이어서 학습할 경우
    model.load_weights('weights.h5')
except:
    pass

# 학습
model.fit(xtrain, ytrain, validation_data=[xtest, ytest],
        epochs=opt.EPOCHS, batch_size=opt.BATCH_SIZE, verbose=1, shuffle=True)

# 모델 저장
model.save_weights('weights.h5')

