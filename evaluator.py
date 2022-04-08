import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


class evaluator:
    def __init__(self, classes=11):
        self.classes = classes

    def set_data(self, label, pred):
        self.label = np.array(label)
        self.pred = np.array(pred)

    def _print_acc(self):
        cm = confusion_matrix(self.label, self.pred)
        print(cm)
        tn, fp, fn, tp = cm.flatten()
        self.accuracy = accuracy_score(self.label, self.pred)
        self.precision = precision_score(self.label, self.pred)
        self.recall = recall_score(self.label, self.pred)
        self.f1 = f1_score(self.label, self.pred)

        print(f"acc: [{self.accuracy}]")
        print(f"precision: [{self.precision}]")
        print(f"recall: [{self.recall}]")
        print(f"f1: [{self.f1}]")

    def print_eval(self, mode=None):
        self._print_acc()
        # self.return_F1_purpuse_score()

    def return_eval_score(self):
        return self.accuracy
