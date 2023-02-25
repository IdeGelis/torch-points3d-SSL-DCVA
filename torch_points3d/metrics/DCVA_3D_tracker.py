import numpy as np
import sklearn.metrics as skmetric
import os
import os.path as osp
import csv
import matplotlib as mpl

mpl.use('Agg')
import time
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from plyfile import PlyData, PlyElement
import torch
import sklearn.metrics as skmetric
from sklearn.neighbors import NearestNeighbors, KDTree

import torch_geometric.transforms as T
from torch_geometric.data import Data

from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models import model_interface
from torch_points3d.metrics.Urb3DCD_deepCluster_tracker import adjusted_rand_score_manual
from torch_geometric.nn.unpool import knn_interpolate
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def read_from_ply(filename, nameInPly="params", sf="label_ch"):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata[nameInPly].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]
        vertices[:, 3] = plydata[nameInPly].data[sf]
    return vertices


class Urb3DCD_dcva_tracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 full_pc: bool = False, full_res: bool = False):
        super(Urb3DCD_dcva_tracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.full_pc = full_pc
        self.full_res = full_res
        self._dataset = dataset

    def reset(self, nb_class=2, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._stage == 'test':
            self._ds = self._dataset.test_data
        elif self._stage == 'val':
            self._ds = self._dataset.val_data
        else:
            self._ds = self._dataset.train_data
        self.nb_class = nb_class
        self._nb_class_ch_orig = self._ds.num_classes_ch
        self.class_color = MplColorHelper('viridis', 0, self.nb_class - 1)
        self._areas = [None] * self._ds.size()
        self._metric_per_areas = [None] * self._ds.size()
        self.gt_tot = None
        self.pred_tot = None
        self.gt_tot_conv = None

    def track(self, model: model_interface.TrackerInterface, data=None, full_pc=False, **kwargs):
        if full_pc:
            inputs = data if data is not None else model.get_layer_data()
            inputs.pred = model.get_output()
            inputs.dmgs_target = model.dmg_pt
            data_l = inputs.to_data_list()
            for p in range(len(data_l)):
                area_sel = data_l[p].area
                # Test mode, compute votes in order to get full res predictions
                if self._areas[area_sel] is None:
                    pair = self._ds._load_save(area_sel)
                    self._areas[area_sel] = pair
                    self._areas[area_sel].prediction_count = torch.zeros(self._areas[area_sel].pos_target.shape[0],
                                                                         dtype=torch.int)
                    self._areas[area_sel].dmg = torch.zeros(self._areas[area_sel].pos_target.shape[0],
                                                            dtype=torch.float)
                    self._areas[area_sel].votes = torch.zeros((self._areas[area_sel].pos_target.shape[0],
                                                               self.nb_class),
                                                              dtype=torch.float)
                    self._areas[area_sel].to(model.device)

                # Gather origin ids and check that it fits with the test set
                if data_l[p].idx_target is None:
                    raise ValueError("The inputs given to the model do not have a idx_target attribute.")

                originids = data_l[p].idx_target
                if originids.dim() == 2:
                    originids = originids.flatten()
                if originids.max() >= self._areas[area_sel].pos_target.shape[0]:
                    raise ValueError("Origin ids are larger than the number of points in the original point cloud.")
                # Set predictions
                self._areas[area_sel].votes[originids] += data_l[p].pred
                self._areas[area_sel].prediction_count[originids] += 1
                self._areas[area_sel].dmg[originids] += data_l[p].dmgs_target

    def finalise(self, save_pc=False, name_test="", saving_path=None, conv_classes=None, filter_pts=0.1,
                 thresholds=None, labels=None, **kwargs):
        gt_tot = []
        gt_tot_conv = []
        pred_tot = []
        print(self._areas)
        if self.full_pc:
            for i, area in enumerate(self._areas):
                if area is not None:
                    # Complete for points that have a prediction
                    area = area.to("cpu")
                    c = ConfusionMatrix(self.nb_class)
                    has_prediction = area.prediction_count > 0
                    pred = torch.argmax(area.votes[has_prediction], 1)
                    pos = area.pos_target[has_prediction]
                    dmgs = area.dmg[has_prediction] / area.prediction_count[has_prediction]
                    # If full res, knn interpolation
                    if self.full_res:
                        _, area_orig_pos, gt = self._ds.clouds_loader(i)
                        # still on GPU no need for num_workers
                        dmgs = knn_interpolate(torch.unsqueeze(dmgs, 1), pos,
                                               area_orig_pos, k=1)
                        if thresholds is None:
                            pred = knn_interpolate(torch.unsqueeze(pred, 1), pos,
                                                   area_orig_pos, k=1).numpy()
                            pred = np.squeeze(pred)
                            pred = pred.astype(int)
                            dmgs = dmgs.numpy()
                        else:
                            pred = self._dmgs_to_labels(dmgs, thresholds, labels=labels).numpy()
                            pred = pred.astype(int)
                            pred = np.squeeze(pred)

                        gt = gt.numpy()
                        pos = area_orig_pos
                        dmgs = np.squeeze(dmgs)
                    else:
                        pred = pred.numpy()
                        gt = area.y[has_prediction].numpy()
                        pos = pos.cpu()
                        dmgs = dmgs.numpy()
                    if conv_classes is not None:
                        gt_conv = np.zeros(gt.shape)
                        for key in conv_classes:
                            gt_conv[gt == key] = conv_classes[key]
                    else:
                        gt_conv = gt

                    if filter_pts>0:
                        pred = self.filter_pred(pred, pos, filter_pts)
                    gt_tot.append(gt)
                    gt_tot_conv.append(gt_conv)
                    pred_tot.append(pred)
                    # Metric computation
                    c.count_predicted_batch(gt_conv, pred)
                    acc = 100 * c.get_overall_accuracy()
                    macc = 100 * c.get_mean_class_accuracy()
                    miou = 100 * c.get_average_intersection_union()
                    class_iou, present_class = c.get_intersection_union_per_class()
                    class_acc = c.confusion_matrix.diagonal() / c.confusion_matrix.sum(axis=1)
                    iou_per_class = {
                        k: "{:.2f}".format(100 * v)
                        for k, v in enumerate(class_iou)
                    }
                    acc_per_class = {
                        k: "{:.2f}".format(100 * v)
                        for k, v in enumerate(class_acc)
                    }
                    miou_ch = 100 * np.mean(class_iou[1:])
                    metrics = {}
                    metrics["{}_acc".format(self._stage)] = acc
                    metrics["{}_macc".format(self._stage)] = macc
                    metrics["{}_miou".format(self._stage)] = miou
                    metrics["{}_miou_ch".format(self._stage)] = miou_ch
                    metrics["{}_iou_per_class".format(self._stage)] = iou_per_class
                    metrics["{}_acc_per_class".format(self._stage)] = acc_per_class

                    self.get_unsup_metric(gt, pred)
                    metrics["{}_RI".format(self._stage)] = self.ri
                    metrics["{}_ARI".format(self._stage)] = self.ari
                    metrics["{}_homogeneity".format(self._stage)] = self.homo
                    metrics["{}_completness".format(self._stage)] = self.compl
                    metrics["{}_v-measure".format(self._stage)] = self.vmeas
                    metrics["{}_MI".format(self._stage)] = self.mi
                    metrics["{}_NMI".format(self._stage)] = self.nmi

                    self._metric_per_areas[i] = metrics
                    print("Result for the whole point cloud number %s in %s : " % (str(i), self._stage))
                    print(metrics)

                    if self._stage == 'test' and save_pc:
                        print('Saving PC %s' % (str(i)))
                        if saving_path is None:
                            saving_path = os.path.join(os.getcwd(), 'res', name_test)
                        if not os.path.exists(saving_path):
                            os.makedirs(saving_path)
                        self._dataset.to_ply(pos, pred,
                                             os.path.join(saving_path, os.path.basename(self._ds.filesPC1[i])[:-4] +
                                                          os.path.dirname(self._ds.filesPC1[i]).split('/')[
                                                              -1] + ".ply"), sf=dmgs
                                             )
            self.gt_tot = np.concatenate(gt_tot)
            self.gt_tot_conv = np.concatenate(gt_tot_conv)
            self.pred_tot = np.concatenate(pred_tot)
            c = ConfusionMatrix(self.nb_class)
            c.count_predicted_batch(self.gt_tot_conv, self.pred_tot)
            acc = 100 * c.get_overall_accuracy()
            macc = 100 * c.get_mean_class_accuracy()
            miou = 100 * c.get_average_intersection_union()
            class_iou, present_class = c.get_intersection_union_per_class()
            iou_per_class = {
                k: "{:.2f}".format(100 * v)
                for k, v in enumerate(class_iou)
            }
            class_acc = c.confusion_matrix.diagonal() / c.confusion_matrix.sum(axis=1)
            acc_per_class = {
                k: "{:.2f}".format(100 * v)
                for k, v in enumerate(class_acc)
            }
            miou_ch = 100 * np.mean(class_iou[1:])
            self.metric_full_cumul = {"acc": acc, "macc": macc, "mIoU": miou, "miou_ch": miou_ch,
                                      "IoU per class": iou_per_class, "acc_per_class": acc_per_class}

            self.get_unsup_metric(self.gt_tot, self.pred_tot)
            self.metric_full_cumul["RI"] = self.ri
            self.metric_full_cumul["ARI"] = self.ari
            self.metric_full_cumul["homogeneity"] = self.homo
            self.metric_full_cumul["completness"] = self.compl
            self.metric_full_cumul["v-measure"] = self.vmeas
            self.metric_full_cumul["MI"] = self.mi
            self.metric_full_cumul["NMI"] = self.nmi

    def get_unsup_metric(self, label, prediction):
        self.ri = skmetric.rand_score(label, prediction)
        self.ari = adjusted_rand_score_manual(label, prediction)
        self.homo, self.compl, self.vmeas = skmetric.homogeneity_completeness_v_measure(label, prediction)
        self.mi = skmetric.mutual_info_score(label, prediction)
        self.nmi = skmetric.normalized_mutual_info_score(label, prediction)
        self._get_repartition(label, prediction)

    def _get_repartition(self, label, prediction):
        self.repartition = np.zeros((self.nb_class, self._nb_class_ch_orig))
        for lab in range(self._nb_class_ch_orig):
            for plab in range(self.nb_class):
                self.repartition[plab, lab] = np.sum((label == lab) & (prediction == plab))

    def _dmgs_to_labels(self, dmg, thresholds, labels=None):
        if labels is None:
            labels = range(len(thresholds) + 1)
        lab = torch.zeros(dmg.shape, dtype=torch.int64)
        for l in range(0, len(labels)):
            if l == 0:
                lab[dmg < thresholds[l]] = labels[l]
            else:
                lab[thresholds[l - 1] <= dmg] = labels[l]
        return lab

    def filter_pred(self, pred, pos, filter_threshold=0.1):
        print("Filtering isolated labeled points")
        tree = KDTree(pos)
        qr = tree.query_radius(pos, r=2)
        for pt in tqdm(range(pred.shape[0])):
            pt_lab = pred[pt]
            if qr[pt].shape[0] > 1:
                neighbors = qr[pt]
                neighbors_lab = pred[neighbors]
                uniques, counts = np.unique(neighbors_lab, return_counts=True)
                counts[uniques == pt_lab] -= 1
                percentages = dict(zip(uniques, counts / (len(neighbors) - 1)))
                if percentages[pt_lab] < filter_threshold:
                    lab = max(percentages, key=percentages.get)
                    pred[pt] = lab

        return pred

    def plot_save_metric_dc(self, plot=False, saving_path=None):
        # Saving DC metrics into a CSV
        if saving_path is None:
            saving_path = os.getcwd()
        with open(osp.join(saving_path, "res.txt"), "w") as fi:
            fi.write("Cumulative full pc res \n")
            for met, val in self.metric_full_cumul.items():
                fi.write(met + " : " + str(val) + "\n")

        if plot:

            pourc = (self.repartition / self.repartition.sum(axis=0)) * 100
            pourc[np.isnan(pourc)] = 0
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.set_xlabel('Ground truth labels')
            index = np.arange(self._nb_class_ch_orig)
            y_offset = np.zeros(self._nb_class_ch_orig)
            for plab in range(self.nb_class):
                ax.bar(index, pourc[plab, :], bottom=y_offset, label=plab,
                       color=self.class_color.get_rgb(plab))
                y_offset += pourc[plab, :]
            lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
            ax.set_xticks(range(self._nb_class_ch_orig))
            ax.set_ylabel("Prediction repartition [%]")
            ax.set_ylim([0, 100])
            fig.savefig(osp.join(saving_path, "repartition.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close(fig)


class simple_tracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, nb_class=0,
                 full_pc: bool = False, full_res: bool = False):
        super(simple_tracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.full_pc = full_pc
        self.full_res = full_res
        self._dataset = dataset
        self.nb_class = nb_class
        self.object_color = np.asarray(
            [
                [81, 109, 114],  # 'ground'   ->  grey
                [81, 163, 148],  # 'vehicle' .-> bluegreen
                [89, 47, 95],  # 'urban furniture' .-> . purple
                [241, 149, 131],  # 'roof'  ->  salmon
                [233, 229, 107],  # 'facade'  ->  yellow
                [77, 174, 84],  # 'shrub'  ->  bright green
                [108, 135, 75],  # 'tree'  ->  dark green
                [79, 79, 76],  # 'vertical surface'   ->  dark grey
                [223, 52, 52],  # 'Chimney'  ->  red
                [79, 79, 76],  # 'table'  ->  dark grey
                [223, 52, 52],  # 'bookcase'  ->  red
                [89, 47, 95],  # 'sofa'  ->  purple

            ]
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._stage == 'test':
            self._ds = self._dataset.test_data
        elif self._stage == 'val':
            self._ds = self._dataset.val_data
        else:
            self._ds = self._dataset.train_data
        self._areas = [None] * self._ds.size()

    def track(self, model: model_interface.TrackerInterface, data=None, full_pc=False, idx_data=0):
        if full_pc:
            inputs = data if data is not None else model.get_input()
            inputs.pred = model.get_output()
            data_l = inputs.to_data_list(target=idx_data)

            for p in range(len(data_l)):
                area_sel = data_l[p].area
                # Test mode, compute votes in order to get full res predictions
                if self._areas[area_sel] is None:
                    pair = self._ds._load_save(area_sel)
                    self._areas[area_sel] = pair
                    if idx_data == 0:
                        self._areas[area_sel].prediction_count = torch.zeros(self._areas[area_sel].pos.shape[0],
                                                                             dtype=torch.int)
                        self._areas[area_sel].votes = torch.zeros((self._areas[area_sel].pos.shape[0],
                                                                   self.nb_class),
                                                                  dtype=torch.float)
                    if idx_data == 1:
                        self._areas[area_sel].prediction_count = torch.zeros(self._areas[area_sel].pos_target.shape[0],
                                                                             dtype=torch.int)
                        self._areas[area_sel].votes = torch.zeros((self._areas[area_sel].pos_target.shape[0],
                                                                   self.nb_class),
                                                                  dtype=torch.float)
                    self._areas[area_sel].to(model.device)

                # Gather origin ids and check that it fits with the test set
                if idx_data == 0:
                    if data_l[p].idx is None:
                        raise ValueError("The inputs given to the model do not have a idx_target attribute.")
                    originids = data_l[p].idx
                    if originids.max() >= self._areas[area_sel].pos.shape[0]:
                        raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

                if idx_data == 1:
                    if data_l[p].idx_target is None:
                        raise ValueError("The inputs given to the model do not have a idx_target attribute.")
                    originids = data_l[p].idx_target
                    if originids.max() >= self._areas[area_sel].pos_target.shape[0]:
                        raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

                if originids.dim() == 2:
                    originids = originids.flatten()
                    # Set predictions
                self._areas[area_sel].votes[originids] += data_l[p].pred
                self._areas[area_sel].prediction_count[originids] += 1

    def finalise(self, save_pc=False, saving_path=None, idx_data=0, **kwargs):
        if self.full_pc:
            for i, area in enumerate(self._areas):
                if area is not None:
                    # Complete for points that have a prediction
                    area = area.to("cpu")
                    has_prediction = area.prediction_count > 0
                    pred = torch.argmax(area.votes[has_prediction], 1)
                    if idx_data == 0:
                        pos = area.pos[has_prediction]

                    if idx_data == 1:
                        pos = area.pos_target[has_prediction]

                    # If full res, knn interpolation
                    if self.full_res:
                        area_orig_pos, area_orig_pos_target, gt = self._ds.clouds_loader(i)
                        # still on GPU no need for num_workers
                        # print(has_prediction.shape)
                        if idx_data == 1:
                            area_orig_pos = area_orig_pos_target
                        pred = knn_interpolate(torch.unsqueeze(pred, 1), pos,
                                               area_orig_pos, k=1).numpy()
                        pred = np.squeeze(pred)
                        pred = pred.astype(int)
                        # gt = gt.numpy()
                        pos = area_orig_pos
                    else:
                        pred = pred.numpy()
                        pos = pos.cpu()

                    if self._stage == 'test' and save_pc:
                        print('Saving PC %s' % (str(i)))
                        if saving_path is None:
                            saving_path = os.path.join(os.getcwd())
                        if not os.path.exists(saving_path):
                            os.makedirs(saving_path)
                        self._dataset.to_ply(pos, pred,
                                             os.path.join(saving_path, os.path.basename(self._ds.filesPC1[i])[:-4] +
                                                          os.path.dirname(self._ds.filesPC1[i]).split('/')[
                                                              -1] + "_" + str(idx_data) + ".ply"),
                                             color=self.object_color
                                             )
