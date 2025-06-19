import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

# 데이터 생성
x = 10 * np.random.rand(50)
y = 2 * x + np.random.randn(50)

# 그래프 그리기
# plt.scatter(x, y, alpha=0.7)
# plt.show()

#  1. estimator class import
# class란? 여기에서는 Linear Regression이 클래스.
# 객체(인스턴스)화화, 즉 우리가 하고자 하는것을 실체화 하기 위해 불러오는 내장된 함수, 코드, 혹은 설계도 같은어떤 것.

from sklearn.linear_model import LinearRegression

# 2. 클래스를 원하는 값으로 인스턴스화해서 모델의 하이퍼파라미터 선택
# 아래 코드를 통해 class는 실제로 사용을 하기 위해 객체화(인스턴스화) 된다
model = LinearRegression(fit_intercept=True)
model
# 위의 fit_intercept=True 는 모델의 하이퍼파라미터 중 하나이다.
# 하이퍼파라미터란 사용자의 활용 방향에 따라 모델 내부에 사용가능한 옵션.

# 3. 특징배열과 대상벡터 배치
X = x[:, np.newaxis]

# 4. 데이터 적합
model.fit(X, y)

model.coef_
model.intercept_

#5. 새 데이터에 대해 모델 적용
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis] 
yfit = model.predict(Xfit)

plt.scatter(xfit, yfit)
plt.scatter(x, y)
plt.show()
