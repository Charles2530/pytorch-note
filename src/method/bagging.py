# Bagging集成方法
# Bagging集成方法是一种基于自助采样的集成方法，它通过对训练数据集进行多次有放回的采样，
# 然后训练多个模型，最后将这些模型的预测结果进行平均或投票来得到最终的预测结果。
import numpy as np


def clone(learner):
    return learner.__class__(**learner.get_params())


class Bagging:
    def __init__(self, base_learner, n_learners):
        self.learners = [clone(base_learner) for _ in range(n_learners)]

    def fit(self, X, y):
        for learner in self.learners:
            bootstrap_indices = np.random.choice(
                X.shape[0], X.shape[0], replace=True)
            learner.fit(X[bootstrap_indices], y[bootstrap_indices])

    def predict(self, X):
        preds = np.array([learner.predict(X) for learner in self.learners])
        return np.mean(preds, axis=0)
