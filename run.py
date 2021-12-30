import numpy as np
import os
from knn import libKNN
from svm import libSVM
from glob import glob
import argparse

def op_load_csv(path):
    f = open(path, 'r')
    content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]
    data = [(x.split(',')[0], int(x.split(',')[1])) for x in content]
    return data

def op_load_npy(path):
    data = np.load(path)
    return data

def op_write_csv(test_files, test_pred_labels, out):
    all_out = []
    all_out.append('id,category')
    for i,each in enumerate(test_files):
        all_out.append('{},{}'.format(each[7:],int(test_pred_labels[i])))
    content = '\n'.join(all_out)
    f = open(out, 'w')
    f.writelines(content)
    f.close()

def op_merge_data(labels, prefix, concat=True):
    """merge data

    Args:
        labels (list(tuple)): labels for each file in csv
        prefix (str): where to load npy
        concat (bool, optional): if concat all data along dimension 0 into one array. Defaults to True.

    Returns:
        label_data (np.ndarray | list(np.ndarray)): labels for all frames, shape [1334*100,] if concat=True
        feats_data (np.ndarray | list(np.ndarray)): feats for all frames, shape [1334*100, 15] if concat=True

    """
    label_data = []
    feats_data = []
    for each in labels:
        npy_data = op_load_npy(os.path.join(prefix, each[0]))
        label_data.append(np.zeros(npy_data.shape[0])+each[1])
        feats_data.append(npy_data)
    if concat:
        label_data = np.concatenate(label_data)
        feats_data = np.concatenate(feats_data)
    return label_data, feats_data

def preprocess_data(label, feats, mode='norm'):
    """preprocess func

    Args:
        mode (str, optional): what you want to do. Defaults to 'norm'.

    Returns:
        label_data (np.ndarray): labels for all frames, shape [1334*100,]
        feats_data (np.ndarray): feats for all frames, shape [1334*100, 15]
    """
    if mode == 'norm':
        N = label.shape[0] if isinstance(label, np.ndarray) else len(label)*label[0].shape[0]
        label_data = label
        max_f = feats.max(axis=1)
        min_f = feats.min(axis=1)
        f_range = max_f - min_f
        feats_data = (feats - feats.min(axis=1).reshape(N,1)) / f_range.reshape(N,1)
        unkeep = np.unique(np.argwhere(np.isnan(feats_data)==True)[:,0])
        keep = np.ones(len(label_data)).astype(np.bool)
        keep[unkeep] = False
    elif mode == 'horizon_norm':
        pass
    else:
        print('error!')
        return None
    return label_data[keep], feats_data[keep], f_range, min_f

def process_test_data(feats, mode='norm'):
    if mode == 'norm':
        N = feats.shape[0]
        max_f = feats.max(axis=1)
        min_f = feats.min(axis=1)
        f_range = max_f - min_f
        feats_data = (feats - feats.min(axis=1).reshape(N,1)) / f_range.reshape(N,1)
        unkeep = np.unique(np.argwhere(np.isnan(feats_data)==True)[:,0])
        keep = np.ones(N).astype(np.bool)
        keep[unkeep] = False
    elif mode == 'horizon_norm':
        pass
    else:
        print('error!')
        return None
    return feats_data[keep]

def cal_label(labels):
    return np.argmax(np.bincount(np.array(labels).astype(np.int32)))

def main(args):
    train_dir = './train/'
    test_dir  = './test/'
    label_train = './label_train.csv'

    # load labels and corresponding feats
    label = op_load_csv(label_train)
    labels,feats = op_merge_data(label, train_dir, concat=True)
    # preprocess
    train_labels, train_feats, _, _ = preprocess_data(labels, feats, mode='norm') #133400 frames, each frame has 15 channels.

    # model init and training
    if args.mode == 'knn':
        classifier = libKNN(n_neighbors=5, algo='auto')
        out_name = args.mode+'_test_results.csv'
    elif args.mode == 'svm':
        classifier = libSVM(C = args.C)
        out_name = args.mode +'_C_eqs_' + str(args.C)+'_test_results.csv'
    elif args.mode == 'xgboost':
        pass
    else:
        raise "not implemented yet"

    classifier.train(train_labels, train_feats)
    # evaluation on training set
    # classifier.eval(train_labels, train_feats)

    # inference for test set
    # test_file_names: all file names, [0023.npy, 1245.pny, ....]
    # test_pred_labels: predicted labels for each file, [0,1,2,0,....]
    # len(test_file_names) == len(test_pred_labels)
    test_file_names = sorted(glob(os.path.join(test_dir,"*")))
    test_pred_labels = []
    for i in test_file_names:
        test_origin_feats = np.load(i)
        test_feats = process_test_data(test_origin_feats)
        labels = classifier.predict_test(test_feats)
        tmp = cal_label(labels)
        test_pred_labels.append(tmp)

    op_write_csv(test_file_names, test_pred_labels, out=out_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="mode", default='knn')
    parser.add_argument("--C", help="svm's C", type=float, default=1.0)

    args = parser.parse_args()
    main(args)