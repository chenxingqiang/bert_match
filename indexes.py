#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2021/4/21 19:45
# @Author  : Chen.xingqiang
"""
import pandas as pd
import argparse
import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter


def args_parser():
    parser = argparse.ArgumentParser(description='Process of make RAW DATA Indexed.')

    parser.add_argument('--data_path',
                        type=str,
                        default='./user_to_item_bert_latest_month_sample/'
                                '2021-04-19', help='data root paths')
    parser.add_argument('--refine', type=bool,
                        default=False,
                        help='whether or not data refine')
    parser.add_argument('--save_path',
                        type=str,
                        default='./data',
                        help='data root paths')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='',
                        help='dataset names')
    parser.add_argument('--date',
                        type=str,
                        default='20210419',
                        help='data date')
    parser.add_argument('--major_name',
                        type=str,
                        default='user_id',
                        help='major part of data ')
    parser.add_argument('--candidate_name',
                        type=str,
                        default='item_id',
                        help='candidates name')
    parser.add_argument('--part',
                        type=int,
                        default=1,
                        help='data part save date')
    parser.add_argument('--part_len',
                        type=int,
                        default=1000000,
                        help='data part save date')
    parser.add_argument('--seq_min_len',
                        type=int,
                        default=5,
                        help='min lenght of seqences in on major id data')

    return parser.parse_args()


def index(data, index_feature):
    """
    :params:data,DataFrame
    :params:features,List
    """
    feature_map = {}
    feature_max_idx = {}
    for feature in index_feature:
        print("indexed feature names:",feature)
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        mapping = dict(zip(lbe.classes_, range(1, len(lbe.classes_) + 1)))
        feature_map[feature] = mapping
    return data, feature_max_idx, feature_map


if __name__ == "__main__":

    args = args_parser()
    major_name = args.major_name
    candidate_name = args.candidate_name
    features = [major_name, candidate_name]

    datalist = []
    for root, dirs, files in os.walk(args.data_path, topdown=False):
        for name in sorted(files):
            print("file reading :",name)
            datalist.append(pd.read_csv(os.path.join(root, name), sep=',', header=None))

    df_data = pd.concat(datalist, axis=0)
    df_data.columns = features

    user_counter = Counter(df_data[major_name])
    df_data["seq_len"] = df_data[major_name].apply(lambda x: user_counter[x])

    df_user_full = df_data[df_data.seq_len >= args.seq_min_len][features]
    df_user_less = df_data[df_data.seq_len < args.seq_min_len][features]

    print("all full samples:\n")
    print(df_user_full.info())
    print("all less samples:\n")
    print(df_user_less.info())

    # 处理不满足条件的样本
    if args.refine:
        df_refine_list = []
        for name, group in df_user_less.groupby(by=major_name):
            for i in range(5 - len(group)):
                group = group.append(group.head(1))
            df_refine_list.append(group)

        df_refine = pd.concat(df_refine_list, axis=0)
        df_refine_data = pd.concat([df_refine, df_user_full], axis=0)
    else:
        df_refine_data = df_user_full

    # check
    for name, group in df_refine_data.groupby(by=major_name):
        if len(group) < args.seq_min_len:
            print(name)
            print(" ERROR : CHECK FAIL ! ")
            exit(1)

    # file name
    file_name = os.path.join(args.save_path,
                             "{0}-{1}-refine-{2}".format(args.date,
                                                         args.dataset_name,
                                                         int(args.refine)))

    dataset_index, feature_maxidx, feature_index = index(df_refine_data, features[1:])
    print(feature_maxidx)

    if args.part:
        max_part_num = int(len(dataset_index)/args.part_len)
        zfill_num = len(str(max_part_num+1))

        for i in range(max_part_num-1):
            part_name = "-part-"+str(i).zfill(zfill_num)+".txt"
            print("saving dataset part : {0}".format(part_name))

            dataset_index.iloc[i*args.part_len:(i+1)*args.part_len,:].to_csv(file_name + part_name,
                                                                         header=None, index=None, sep=' ')

        print("saving dataset last part : {0}".format("-part-" + str(max_part_num-1).zfill(zfill_num)))
        dataset_index.iloc[(max_part_num-1) * args.part_len:, :].to_csv(
                file_name + "-part-" + str(max_part_num-1).zfill(zfill_num) + ".txt",
                header=None, index=None, sep=' ')

    print("saving dataset all : {0}-all.txt".format(file_name))
    dataset_index.to_csv(file_name + "-all.txt", header=None, index=None, sep=' ')
    print(" ... dataset indexed saving done")

    print("start index saving ...")
    df_index_list = []
    for col in [candidate_name]:
        df = pd.DataFrame(feature_index[col], index=['index']).T.reset_index()
        df["item_value"] = df['level_0'].map(lambda x: col + "#" + str(x))
        df_index_list.append(df[['item_value', 'index']])
    print("saving dataset index : {0}-all-index.txt".format(file_name))
    df_index = pd.concat(df_index_list, axis=0)
    df_index.to_csv(file_name + "-all-index.txt", index=None, sep=',')

    print(" ... index saving done ")
