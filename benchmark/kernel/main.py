from itertools import product

import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gcn import GCN, GCNWithJK
from graph_sage import GraphSAGE, GraphSAGEWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from graclus import Graclus
from top_k import TopK
from diff_pool import DiffPool
from sag_pool import SAGPool
from global_attention import GlobalAttentionNet
from set2set import Set2SetNet
from sort_pool import SortPool

import time

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2, 3, 4, 5]
hiddens = [16, 32, 64, 128]



datasets = ['AIDS',
            'NCI1',
            'BZR',
            'BZR_MD',
            'COIL-DEL',
            'COIL-RAG',
            'COLLAB',
            #'COLORS-3',
            'COX2',
            'COX2_MD',
            'Cuneiform',
            'DBLP_v1',
            'DHFR',
            'DHFR_MD',
            'ER_MD',
            'DD',
            'ENZYMES',
            'Fingerprint',
            'FIRSTMM_DB',
            'FRANKENSTEIN',
            'IMDB-BINARY',
            'IMDB-MULTI',
            'KKI',
            'Letter-high',
            'Letter-low',
            'Letter-med',
            'MCF-7',
            'MCF-7H',
            'MOLT-4',
            'MOLT-4H',
            'Mutagenicity',
            'MSRC_9',
            'MSRC_21',
            'MSRC_21C',
            'MUTAG',
            'NCI1',
            'NCI109',
            'NCI-H23',
            'NCI-H23H',
            'OHSU',
            'OVCAR-8',
            'OVCAR-8H',
            'P388',
            'P388H',
            'PC-3',
            'PC-3H',
            'Peking_1',
            'PTC_FM',
            'PTC_FR',
            'PTC_MM',
            'PTC_MR',
            'PROTEINS',
            'PROTEINS_full',
            'REDDIT-BINARY',
            'REDDIT-MULTI-5K',
            'REDDIT-MULTI-12K',
            'SF-295',
            'SF-295H',
            'SN12C',
            'SN12CH',
            'SW-620',
            'SW-620H',
            'SYNTHETIC',
            'SYNTHETICnew',
            'Synthie',
            'Tox21_AHR',
            'Tox21_AR',
            'Tox21_AR-LBD',
            'Tox21_ARE',
            'Tox21_aromatase',
            'Tox21_ATAD5',
            'Tox21_ER',
            'Tox21_ER_LBD',
            'Tox21_HSE',
            'Tox21_MMP',
            'Tox21_p53',
            'Tox21_PPAR-gamma',
            'TWITTER-Real-Graph-Partial',
            'UACC257',
            'UACC257H',
            'Yeast',
            'YeastH']




nets = [
    GCNWithJK,
    GraphSAGEWithJK,
    GIN0WithJK,
    GINWithJK,
    Graclus,
    TopK,
    SAGPool,
    DiffPool,
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))


dataset_names = []
Nets = []
df_num_layers = []
df_hidden = []
df_loss = []
df_acc = []
df_std = []
df_iterations_time = []


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    #print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        start_time = time.time()
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None)
        if loss < best_result[0]:
            best_result = (loss, acc, std)

        dataset_names.append(dataset_name)
        Nets.append(Net)
        df_num_layers.append(num_layers)
        df_hidden.append(hidden)
        df_loss.append(loss)
        df_acc.append(acc)
        df_std.append(std)
        iteration_time = time.time()
        df_iterations_time.append(iteration_time - start_time)
        df = pandas.DataFrame(data={"Dataset name": dataset_names, "Network": Nets, "Num layers": df_num_layers, "Hidden layer size": df_hidden, "Accuracy": df_acc, "Std": df_std, "Loss": df_loss, "Iteration time": df_iterations_time})
        df.to_csv("standard_benchmarks.csv", sep=',',index=False)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    #print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
