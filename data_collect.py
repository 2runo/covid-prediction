"""
코로나 신규 확진자(Daily new cases) 데이터를 크롤링한다.
정보 제공 사이트 : https://www.worldometer.info
"""
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def get_data(country):
    # 특정 country의 코로나 신규 확진자 데이터를 크롤링한다.
    # 사이트 : www.worldometers.info
    req = urllib.request.Request('https://www.worldometers.info/coronavirus/country/%s/' % country,
                                 headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req).read().decode('utf-8')

    data = eval(raw.split('Daily Cases')[-1].split('data: ')[1].split('}')[0].replace(' ', '').replace('null', '0'))  # 신규 확진자 수
    data = np.array([i if i >= 0 else 0 for i in data])  # 음수는 0 처리 (이상치 처리)
    return data


def upper_half(a):
    # 두 반환값의 합이 a가 되도록 한다.
    half = a // 2
    if a % 2 == 0:
        return half, half
    return half, half + 1


def moving_average(data, size=2):
    # 이동 평균을 계산하여 반환한다.
    result = []
    size = upper_half(size)
    for i in range(len(data)):
        result.append(np.mean(data[i-size[0]:i+size[1]]))
    return result


def rescale(data, crit, measure="mean", cutratio=0.):
    # data를 crit과 비슷한 값으로 rescaling한다.
    if measure == "max":
        # 최댓값이 비슷해지도록 rescaling한다.
        crit = crit ** np.random.uniform(1.05, 1.07)
        measure_fun = lambda data, crit: np.linalg.norm(max(data) - max(crit))
    elif measure == "median":
        # 최댓값이 비슷해지도록 rescaling한다.
        # 단, crit의 앞 부분은 무시하고 최댓값을 구한다.
        crit = crit ** 1.1
        measure_fun = lambda data, crit: np.linalg.norm(max(data) - max(crit[75:]))

    w = min([[measure_fun(data * i, crit), i] for i in np.arange(0, 4, 0.0001)])[1]
    return data * w


# 크롤링할 국가들 목록
countries = ['us', 'brazil', 'india', 'russia', 'south-africa', 'peru', 'mexico', 'colombia', 'spain', 'chile', 'iran',
             'argentina', 'uk', 'saudi-arabia', 'bangladesh', 'pakistan', 'italy', 'turkey', 'france', 'germany',
             'iraq', 'philippines', 'indonesia', 'canada', 'qatar', 'bolivia', 'ecuador', 'ukraine', 'kazakhstan',
             'israel', 'egypt', 'dominican-republic', 'panama', 'sweden', 'oman', 'belgium', 'kuwait', 'romania',
             'belarus', 'guatemala', 'united-arab-emirates', 'netherlands', 'poland', 'japan', 'singapore', 'portugal',
             'honduras', 'morocco', 'nigeria', 'bahrain', 'ghana', 'kyrgyzstan', 'armenia', 'algeria', 'ethiopia',
             'switzerland', 'venezuela', 'uzbekistan', 'afghanistan', 'azerbaijan', 'costa-rica', 'moldova', 'kenya',
             'nepal', 'serbia', 'ireland', 'austria', 'australia', 'el-salvador', 'czech-republic', 'state-of-palestine',
             'cameroon', 'bosnia-and-herzegovina', 'cote-d-ivoire', 'denmark', 'bulgaria', 'madagascar', 'macedonia',
             'paraguay', 'senegal', 'sudan', 'lebanon', 'zambia', 'libya', 'norway']

# 대한민국부터 크롤링
kr_data = get_data('south-korea')

# countries의 국가들 차례대로 크롤링
for country in countries:
    print('country:', country)

    cur_data = get_data(country)  # 신규 확진자 데이터 크롤링

    result = []

    cur_data2 = rescale(cur_data, kr_data, measure="median")  # 최댓값 기준으로 rescaling (y축 rescaling)
    cur_data2 = moving_average(cur_data2, size=4)  # 이동 평균 -> 들락날락한 데이터 제거
    result.append(cur_data2)

    cur_data = rescale(cur_data, kr_data, measure="max")  # 최댓값 기준으로 rescaling (y축 rescaling)
    cur_data = resize(cur_data.reshape((-1, 1)), (int(len(cur_data)*0.7), 1)).reshape(-1)  # x축 rescaling
    cur_data = moving_average(cur_data, size=4)  # 이동 평균 -> 들락날락한 데이터 제거
    result.append(cur_data)

    # 데이터 저장
    with open('data/%s_data.txt' % country, 'w', encoding='utf8') as f:
        f.write('\n'.join([' '.join(map(str, i)) for i in result]))

# 가장 마지막 국가의 데이터 시각화
plt.plot(kr_data, color='b')
plt.plot(cur_data, color='r')
plt.plot(cur_data2, color='g')
plt.legend(['korea', country + '(median)', country + '(max)'])
plt.show()
