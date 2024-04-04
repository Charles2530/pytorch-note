# Stacking集成方法
# Stacking集成方法是一种集成学习方法，它通过将多个基学习器的预测结果作为输入，再训练一个学习器来预测最终的输出结果。
# Stacking集成方法的基本思想是：将多个基学习器的预测结果作为输入，再训练一个学习器来预测最终的输出结果。
# Stacking集成方法的优点是：可以将多个基学习器的优点进行整合，从而得到更好的预测结果。
# Stacking集成方法的缺点是：需要更多的计算资源，因为需要训练多个基学习器。
import numpy as np


class Stacking:
    def __init__(self, base_learners, meta_learner):
        self.base_learners = base_learners
        self.meta_learner = meta_learner

    def fit(self, X, y):
        base_learners_preds = np.array(
            [learner.predict(X) for learner in self.base_learners]).T
        self.meta_learner.fit(base_learners_preds, y)

    def predict(self, X):
        base_learners_preds = np.array(
            [learner.predict(X) for learner in self.base_learners]).T
        return self.meta_learner.predict(base_learners_preds)
