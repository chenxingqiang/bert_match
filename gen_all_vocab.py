# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
from util import *
from vocab import *
import pickle

random_seed = 12345
short_seq_prob = 0
# Probability of creating sequences which are shorter than the maximum length。

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("signature", 'default', "signature_name")

flags.DEFINE_string(
    "data_dir", './data/',
    "data dir.")

flags.DEFINE_string(
    "dataset_name", 'ml-1m',
    "all dataset name.")

def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    output_dir = FLAGS.data_dir
    dataset_name = FLAGS.dataset_name
    version_id = FLAGS.signature
    print(version_id)

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    dataset = data_partition(output_dir+dataset_name+'.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
        format(
        len(user_train),
        len(user_valid), len(user_test), usernum, itemnum))

    for idx, u in enumerate(user_train):
        if idx <10:
            print("##### user test ########")
            print(user_train[u])
            print(print("##### user valid ########"))
            print(user_valid[u])
            print("######## user_test #########")
            print(user_test[u])

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')


if __name__ == "__main__":
    main()
