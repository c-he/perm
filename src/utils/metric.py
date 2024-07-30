from typing import Dict

import numpy as np
import pandas as pd
import torch
from pandas import ExcelWriter
from sklearn import metrics

from utils.misc import EPSILON, flatten_list


def compute_labeling_metrics(labels_gt, labels_rec, create_excel_dfs=True, out_fname=None):
    labels_gt = flatten_list(labels_gt)
    labels_rec = flatten_list(labels_rec)

    assert len(labels_rec) != 0

    # superset_labels = sorted(list(set(labels_gt)))
    superset_labels = sorted(list(set(labels_rec + labels_gt)))
    # if 'nan' not in superset_labels:
    #     superset_labels += ['nan']
    # else:
    #     superset_labels.pop(superset_labels.index('nan'))
    #     superset_labels += ['nan']

    all_label_map = {k: superset_labels.index(k) for k in superset_labels}
    assert len(all_label_map) == len(set(all_label_map.keys()))  # keys should be unique
    #
    label_ids_gt = np.array([all_label_map[k] for k in labels_gt])
    label_ids_rec = np.array([all_label_map[k] for k in labels_rec])

    avg_mode = 'macro'

    # The support is the number of occurrences of each class in y_true.
    # so if a label is not present in the labels_gt but present in labels_rec it will get a 0 percent.
    # this could happen when a maker layout is changed for a capture and soma still assigns a nearby label.
    labeling_report = metrics.classification_report(label_ids_gt, label_ids_rec,
                                                    output_dict=True, labels=np.arange(len(superset_labels)),
                                                    target_names=superset_labels, zero_division=0)

    # accuracy = accuracy_score(label_ids_gt, label_ids_rec)
    # accuracy = jaccard_score(label_ids_gt, label_ids_rec, labels=np.arange(len(superset_labels)), average='macro')
    #
    f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
    precision = labeling_report[f'{avg_mode} avg']['precision']
    recall = labeling_report[f'{avg_mode} avg']['recall']
    accuracy = labeling_report['accuracy']

    results = {'f1': f1_score,
               'acc': accuracy,
               'prec': precision,
               'recall': recall
               }

    if create_excel_dfs:
        cm = metrics.confusion_matrix(label_ids_gt, label_ids_rec, labels=range(len(superset_labels)))

        # per_class_acc = cm.diagonal()/cm.sum(axis=1)
        # for k, v in zip(superset_labels, per_class_acc):
        #     labeling_report[k].update({'acc':v})

        df_cm = pd.DataFrame(cm, index=superset_labels, columns=superset_labels)

        labeling_report = pd.DataFrame(labeling_report).transpose()

        excel_dfs = {'labeling_report': labeling_report,
                     'confusion_matrix': df_cm}
        results.update(excel_dfs)

        if out_fname:
            assert out_fname.endswith('.xlsx')
            save_xlsx(excel_dfs, xlsx_fname=out_fname)

    return results


def save_xlsx(dicts_dfs, xlsx_fname):
    with ExcelWriter(xlsx_fname, engine='xlsxwriter') as writer:
        for name, df in dicts_dfs.items():
            df.to_excel(writer, sheet_name=name)


def clustering_metrics(X, labels):
    silhouette_score = metrics.silhouette_score(X, labels)
    ch_index = metrics.calinski_harabasz_score(X, labels)
    db_index = metrics.davies_bouldin_score(X, labels)

    return {'silhouette_score': silhouette_score,
            'ch_index': ch_index,
            'db_index': db_index
            }


def curvature(position: torch.Tensor) -> torch.Tensor:
    # we first compute the circumradius r for every 3 adjacent points on the strand, and curvature is defined as 1/r.
    # https://en.wikipedia.org/wiki/Circumscribed_circle
    a = position[..., :-2, :] - position[..., 2:, :]  # (..., num_samples - 2, 3)
    b = position[..., 1:-1, :] - position[..., 2:, :]  # (..., num_samples - 2, 3)
    c = a - b
    curvature = 2.0 * torch.norm(torch.cross(a, b, dim=-1), dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) * torch.norm(c, dim=-1) + EPSILON)  # (batch_size, num_strands, num_samples - 2)

    return curvature


def export_csv(csv_fname: str, metrics: Dict) -> None:
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(csv_fname)
