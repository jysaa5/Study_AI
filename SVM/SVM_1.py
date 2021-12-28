# <선형 SVM 분류>
# 붓꽃 데이터셋을 적재하고 특성 스케일을 변경하고 Iris-Virginica 품종을 감지하기 위해 선형 SVM 모델을 훈련시킨다.
# C=1, 힌지 손실(hinge loss)함수를 적용한 LinearSVC클래스를 사용한다.

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x = iris["data"][:, (2,3)] # 꽃잎 길이, 꽃잎 너비
y = (iris["target"] == 2).astype(np.float64) #Iris-Virginica

# StandardScaler: SVM 특성의 스케일을 조정
# C: 제약조건의 강도를 설정함. -> C가 낮으면 마진이 크게 잡히고 C가 높으면 마진이 작게 잡힌다.
svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])

# x: 꽃잎 길이, 꽃잎 너비
# y: 0.0 과 1.0으로 분류함.
svm_clf.fit(x, y)

# 예측
print(svm_clf.predict([[5.5, 1.7]]))
