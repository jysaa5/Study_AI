# 3차 다항식 커널을 사용해 SVM분류기를 훈련시킨다.

from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

x, y = make_moons(n_samples=100, noise=0.15)
poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel="poly"
                                                ,degree=3
                                 ,coef0=1, C=5))])

poly_kernel_svm_clf.fit(x, y)