import torch
import torch.nn as nn


def build_loss_func(config: dict, weight: list = None):
    # criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = Ce_Bce_Loss(config.pairs, weight)
    criterion = MultiDiffLoss()

    return criterion


class MultiDiffLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._diff_weight = 0.2
        self._src_weight = 0.2

    def forward(self, outputs, labels):
        total_loss = self._bce_loss(outputs["out"], labels)
        total_loss += (
            self._bce_loss(outputs["aero_diff_out"], labels) * self._diff_weight
        )
        total_loss += (
            self._bce_loss(outputs["ele_diff_out"], labels) * self._diff_weight
        )
        total_loss += self._bce_loss(outputs["aero_src_out"], labels) * self._src_weight
        total_loss += self._bce_loss(outputs["ele_src_out"], labels) * self._src_weight
        return total_loss
