from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_geometric.nn import knn
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch
log = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class KPConv_contrastive(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):

        # Extract parameters from the dataset
        self._num_classes = option.define_constants.num_pseudo_label
        self._weight_classes = dataset.weight_classes
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.I1 = option.define_constants.I1

        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            self.FC_layer.add_module("Class", Lin(in_feat,self._num_classes, bias=False))

        self.loss_names = ["loss_seg"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]
        self.lossFctSecondary = torch.nn.L1Loss()
        # self.similarity =  torch.nn.CosineSimilarity(dim=1, eps=1e-08)


    def set_input(self, datas, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        if self.training:
            data = datas[0]
            self.input1, self.input2 = data.to_data()
            self.input1 = SimpleBatch(pos=self.input1.pos, batch=self.input1.batch)
            self.input2 = SimpleBatch(pos=self.input2.pos, batch=self.input2.batch, y=data.y)

            data_s = datas[1] # Shuffled data
            _, self.input2_s = data_s.to_data()

            self.input2_s = SimpleBatch(pos=self.input2_s.pos, batch=self.input2_s.batch, y=data_s.y)

            self.input1 = self.input1.to(device)
            self.input2 = self.input2.to(device)
            self.input2_s = self.input2_s.to(device)

            self.input1.x = add_ones(self.input1.pos, self.input1.x, True)
            self.input2.x = add_ones(self.input2.pos, self.input2.x, True)
            self.input2_s.x = add_ones(self.input2_s.pos, self.input2_s.x, True)

        else:
            self.input = datas
            self.input = self.input.to(device)
            self.input.x = add_ones(self.input.pos, self.input.x, True)
        self.upsample = None
        self.pre_computed = None


    def forward_kpfcnn(self, data) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, precomputed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)

            data = self.inner_modules[0](data)
            innermost = True
        self.inner_data = data
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample)

        last_feature = data.x
        if self._use_category:
            self.output = self.FC_layer(last_feature, self.category)
        else:
            self.output = self.FC_layer(last_feature)
        return self.output, self.inner_data

    def forward_dcva(self, id_layer_sel = [], compute_loss=False, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []
        self.layers = []
        data = self.input
        id_layer = 1
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=self.pre_computed)
            stack_down.append(data)
            if id_layer in id_layer_sel:
                self.layers.append(data)
            id_layer += 1

        data = self.down_modules[-1](data, precomputed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            if id_layer in id_layer_sel:
                self.layers.append(data)
            id_layer += 1
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
                if id_layer in id_layer_sel:
                    self.layers.append(data)
                id_layer += 1
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample)
                if id_layer in id_layer_sel:
                    self.layers.append(data)
                id_layer += 1

        last_feature = data.x
        if self._use_category:
            self.output = self.FC_layer(last_feature, self.category)
        else:
            self.output = self.FC_layer(last_feature)
        if id_layer in id_layer_sel:
            data.x = self.output
            self.layers.append(data)
        if compute_loss:
            if self.labels is not None:
                self.compute_loss()

        self.data_visual = self.input
        self.data_visual.pred = torch.max(self.output, -1)[1]
        return self.output

    def forward(self, epoch=0, iteration=0, compute_loss=True, id_layer_sel = [], *args, **kwargs) -> Any:
        if self.training:
            self.y1, self.y1_innerdata = self.forward_kpfcnn(self.input1)
            self.y2, self.y2_innerdata = self.forward_kpfcnn(self.input2)
            self.y2_s, self.y2_s_innerdata = self.forward_kpfcnn(self.input2_s)
            if compute_loss:
                self.compute_loss(epoch, iteration)
        else:
            self.forward_dcva(id_layer_sel)


    def compute_loss(self, epoch, iteration):
        _, pred1 = torch.max(self.y1, 1)
        _, pred2 = torch.max(self.y2, 1)

        # weights computation for the loss ponderation
        count = torch.bincount(torch.cat((pred1, pred2, torch.arange(0,self._num_classes).to(self.output.device))))
        count = count.float()
        count = torch.sqrt(torch.mean(count) / count)
        weights = count / torch.sum(count)
        weights = weights.to(self.output.device)
        # L1
        self.lossPrimary1 = F.cross_entropy(self.y1, pred1, weight = weights)
        # L2
        self.lossPrimary2 = F.cross_entropy(self.y2, pred2, weight = weights)

        self.lossPrimary = (self.lossPrimary1 + self.lossPrimary2) / 2


        nn_list = knn(self.input1.pos, self.input2.pos, 1, self.input1.batch, self.input2.batch)
        self.lossSecondary1 = self.lossFctSecondary(self.y1[nn_list[1, :], :], self.y2)
        # L1,2'
        nn_list = knn(self.input1.pos, self.input2_s.pos, 1, self.input1.batch, self.input2_s.batch)
        self.lossSecondary2 = -self.lossFctSecondary(self.y1[nn_list[1, :], :], self.y2_s)
        self.lossSecondary2 = self.lossSecondary2.exp()

        if epoch <= self.I1:
            self.loss = self.lossPrimary
        else:
            if iteration % 2 == 0:
                self.loss = self.lossPrimary
            elif iteration % 2 == 1:
                self.loss = self.lossSecondary1.mean()
            elif iteration % 3 == 2:
                self.loss = self.lossSecondary2.mean()


    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()

    def optimize_parameters(self, epoch, batch_size, iteration=0):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        with torch.cuda.amp.autocast(enabled=self.is_mixed_precision()):  # enable autocasting if supported
            self.forward(iteration=iteration, epoch=epoch)  # first call forward to calculate intermediate results

        orig_losses = self._do_scale_loss()  # scale losses if needed
        make_optimizer_step = self._manage_optimizer_zero_grad()  # Accumulate gradient if option is up
        self.backward()  # calculate gradients
        self._do_unscale_loss(orig_losses)  # unscale losses to orig

        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self._grad_clip)

        if make_optimizer_step:
            self._grad_scale.step(self._optimizer)  # update parameters

        if self._lr_scheduler:
            self._do_scheduler_update("_update_lr_scheduler_on", self._lr_scheduler, epoch, batch_size)

        if self._bn_scheduler:
            self._do_scheduler_update("_update_bn_scheduler_on", self._bn_scheduler, epoch, batch_size)

        self._grad_scale.update()  # update scaling
        self._num_epochs = epoch
        self._num_batches += 1
        self._num_samples += batch_size
