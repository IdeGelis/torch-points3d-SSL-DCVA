import os
import os.path as osp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torchnet as tnt
import torch
from typing import Dict, Any
import wandb
from torch.utils.tensorboard import SummaryWriter
import logging

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


def meter_value(meter, dim=0):
    return float(meter.value()[dim]) if meter.n > 0 else 0.0


class selfsup_tracker(BaseTracker):
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool):
        super(selfsup_tracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)
        self._metric_func = {
                            "l1_loss": min,
                            "l2_loss": min,
                            "l12_loss": min,
                            "l12cont_loss":min,

                            }

        self.lossPrimary1Array = torch.empty((1))
        self.lossPrimary2Array = torch.empty((1))
        self.lossSecondary1Array = torch.empty((1))
        self.lossSecondary2Array = torch.empty((1))

    def reset(self, stage="train"):
        super().reset(stage=stage)


    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor


    def track(self, model: model_interface.TrackerInterface, conv_classes=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)
        self._l1_loss = model.lossPrimary1.item()
        self._l2_loss = model.lossPrimary2.item()
        self._l12_loss = model.lossSecondary1.item()
        self._l12cont_loss = model.lossSecondary2.item()

        self.lossPrimary1Array = torch.cat((self.lossPrimary1Array, model.lossPrimary1.unsqueeze(0).cpu().detach()))
        self.lossPrimary2Array = torch.cat((self.lossPrimary2Array, model.lossPrimary2.unsqueeze(0).cpu().detach()))
        self.lossSecondary1Array = torch.cat((self.lossSecondary1Array, model.lossSecondary1.unsqueeze(0).cpu().detach()))
        self.lossSecondary2Array = torch.cat((self.lossSecondary2Array, model.lossSecondary2.unsqueeze(0).cpu().detach()))


    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_l1_loss".format(self._stage)] = self._l1_loss
        metrics["{}_l2_loss".format(self._stage)] = self._l2_loss
        metrics["{}_l12_loss".format(self._stage)] = self._l12_loss
        metrics["{}_l12cont_loss".format(self._stage)] = self._l12cont_loss
        return metrics

    @property
    def metric_func(self):
        return self._metric_func

    def finalise(self, plot_loss = True, **kwargs):
        if plot_loss:
            plt.figure()
            plt.plot(self.lossPrimary1Array[1:], label="L1")
            plt.plot(self.lossPrimary2Array[1:], label="L2")
            plt.plot(self.lossSecondary1Array[1:], label="L12")
            plt.plot(self.lossSecondary2Array[1:], label="L12_contrastive")
            plt.xlabel("Iteration")
            plt.legend()
            plt.grid()
            plt.savefig(osp.join(os.getcwd(), "losses.png"))
            plt.close()

