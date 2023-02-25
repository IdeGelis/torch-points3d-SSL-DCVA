from typing import Any
from omegaconf import OmegaConf
import copy
import torch
import torch.nn.functional as F

from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch


import torch_points3d.models.segmentation.pointnet_yanx27 as pointnet

def dist(pt1, pt2):
    return torch.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

class DCVA_3D(BaseModel):
    def __init__(self,opt, model_type, dataset, modules):
        opt = copy.deepcopy(opt)
        super(DCVA_3D, self).__init__(opt)
        self.opt = opt
        self.knn = int(self.opt.knn)
        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self.opt.pretrained_model.path_pretrained_model,
            self.opt.pretrained_model.model_name,
            self.opt.pretrained_model.weight_name,
            resume=True,
        )

        self._checkpoint.dataset_properties = OmegaConf.create(self._checkpoint.dataset_properties)
        if not self._checkpoint.is_empty:
            self._model_pre_trained: BaseModel = self._checkpoint.create_model(
                self._checkpoint.dataset_properties, weight_name=self.opt.pretrained_model.weight_name
            )
        else:
            raise ValueError("Pretrained model {} does not exist.".format(self.opt.pretrained_model.path_pretrained_model))
        self.conv_type = self._model_pre_trained.conv_type
        self.layer_select = [self.opt.layer_for_dcva]
        self.dmgs = None
        self.layer_data = None

    def set_input(self, data, device):
        self.input0, self.input1 = data.to_data()
        self.input0 = SimpleBatch(pos=self.input0.pos, batch=self.input0.batch, area = self.input0.area, idx=self.input0.idx)
        self.input1 = SimpleBatch(pos=self.input1.pos, batch=self.input1.batch, area = self.input0.area, idx=self.input1.idx,
                            y=data.y)

    def set_layer_data(self,data):
        self.layer_data = data

    def get_layer_data(self):
        if self.layer_data is None:
            self.get_input()
        else:
            return self.layer_data

    def get_dmgs(self, save_dmg = False, thresholds = None, labels = None ) -> Any:
        data0 = self.input0
        data1 = self.input1
        self._model_pre_trained.eval()
        with torch.no_grad():
            self._model_pre_trained.set_input(data1,self.device)
            with torch.cuda.amp.autocast(enabled=self._model_pre_trained.is_mixed_precision()):
                self._model_pre_trained.forward(id_layer_sel =self.layer_select, compute_loss = False)
            data1_layer = self._model_pre_trained.layers

            self._model_pre_trained.set_input(data0, self.device)
            with torch.cuda.amp.autocast(enabled=self._model_pre_trained.is_mixed_precision()):
                self._model_pre_trained.forward(id_layer_sel =self.layer_select, compute_loss = False)
            data0_layer = self._model_pre_trained.layers

            assert len(data1_layer)>0 and len(data0_layer)>0

            # Get difference vector of selected layer
            if self.opt.pretrained_model.model_name == "KPConvPaper" or self.opt.pretrained_model.model_name == "KPConv_contrastive":
                diff_l = []
                for l in range(len(data1_layer)):
                    d0_l = data0_layer[l]
                    d1_l = data1_layer[l]
                    nn_list = knn(d0_l.pos[:,:3], d1_l.pos[:,:3], self.knn, d0_l.batch, d1_l.batch)
                    diff = torch.abs(d1_l.x[nn_list[0,:],:] - d0_l.x[nn_list[1,:],:])
                    diff = diff.reshape((d1_l.x.shape[0],self.knn, d1_l.x.shape[1]))
                    # to take the minimal difference if several nearest neighbors are used
                    # (it was implemented to test if the results was better but finally no, so we let self.knn=1 (see in the conf file)
                    val, ind = torch.min(diff, dim=1)
                    diff_l.append(val)

            else:
                diff_l = []
                for l in range(len(data1_layer)):
                    d0_l = data0_layer[l]
                    d1_l = data1_layer[l]
                    nn_list = knn(data0.pos[:,:3], data1.pos[:,:3], 1, data0.batch, data1.batch)
                    diff_l.append(torch.abs(d1_l - d0_l[nn_list[1, :], :]))
            # Constrution of hyper-vector G
            G = torch.cat(diff_l, dim=1)

            # Get deep magnitude coefficient
            dmg = torch.norm(G, dim=1, p=2) #Frobenius norm
            self.dmg_pt = dmg
            if save_dmg:
                if self.dmgs is None:
                    self.dmgs = dmg
                else:
                    self.dmgs = torch.cat((self.dmgs,dmg))
            if thresholds is not None:
                lab = self._dmgs_to_labels(dmg=dmg, thresholds = thresholds, labels=labels)
                self.output = F.one_hot(lab, num_classes = max(labels)+1).to(self.device)
            if self.opt.pretrained_model.model_name == "KPConvPaper" or self.opt.pretrained_model.model_name == "KPConv_contrastive":
                data = PairBatch(batch=data0_layer[0].batch, batch_target=data1_layer[0].batch,
                                 pos=data0_layer[0].pos, pos_target=data1_layer[0].pos,
                                 idx=data0_layer[0].idx, idx_target=data1_layer[0].idx,
                                 area=data1_layer[0].area, y=data1_layer[0].y)
            else:
                data = PairBatch(batch=data0.batch, batch_target=data1.batch,
                                 pos=data0.pos, pos_target=data1.pos,
                                 idx=data0.idx, idx_target=data1.idx,
                                 area=data1.area, y=data1.y)

            self.set_layer_data(data)


    def _dmgs_to_labels(self, dmg, thresholds, labels=None):
        if labels is None:
            labels = range(len(thresholds) + 1)
        lab = torch.zeros(dmg.shape, dtype=torch.int64)
        for l in range(0,len(labels)):
            if l == 0:
                lab[dmg <thresholds[l]] = labels[l]
            else:
                lab[thresholds[l - 1] <= dmg] = labels[l]
        return lab


