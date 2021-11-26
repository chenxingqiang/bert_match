#  基于 BERT 模型的个性化召回

### 数据构建准备过程测试
#### step 1 : 切割数据
      $ python indexes.py --data_path /home/chenxingqiang/bert_match/bert_latest_month_sample/20210419 --save_path ./data/ --date 20210419
      $  nohup python indexes.py --data_path /home/chenxingqiang/bert_match/bert_latest_month_sample/20210419 --save_path ./data/ --date 20210419 >> 20210419_indexes.log 2>&1

#### step 2 : 生成全局词表
      $ ./run_gen_all_vocab.sh task-refine-0-all
 
#### step 3 : 单次生成 tfrecord 训练样本
      $ ./run_genpart_dataset.sh task-refine-0-part-000 20210419-task-refine-0-all 

#### step 4 : 批量生成 tfrecord 训练样本
      $ ./run_genpart_dataset_batch.sh 20210419-task-refine-0-part-00 20210419-task-refine-0-all  1 2
      $ nohup ./run_genpart_dataset_batch.sh 20210419-task-refine-0-part- 20210419-task-refine-0-all  0 134 >> run_genpart_dataset_batch.log 2>&1 &
      
### step 5 :  单次训练 
      $ ./run_train_eval.sh 20210419-task-refine-0-part-000 20210419-task-refine-0-all 1 0   
      $ nohup ./run_train_eval.sh 20210419-task-refine-0-part-000 20210419-task-refine-0-all 1 0 >> run_train_eval_20210419-task-refine-0-part-000.log 2>&1 &
      $ nohup ./run_train_eval_batch.sh 20210419-task-refine-0-all 100 >> ./run_train_eval_batch.log 2>&1
### 训练数据和测试数据
```yaml
{'masked_lm_positions': 

        array([48,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0]), 
'info': 
          array([777800]), 

'masked_lm_ids': 
            array([182535,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0]), 

'masked_lm_weights': 
      array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0.], dtype=float32), 

'input_mask': 
       array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),

 'input_ids': 
        array([182535, 182535, 182535, 182535, 182535, 182535,   1991, 182535,
       182535, 182535, 182535, 182535, 182535, 182535, 182535, 182535,
       182535, 182535, 182535, 182535,    700,    700, 182535, 182535,
       182535, 182535, 182535, 182535, 182535, 182535, 182535, 182535,
       182535, 182535, 182535, 182535, 182535,   1991, 182535, 182535,
       182535,   1991, 182535, 182535, 182535, 182535, 182535, 182535,
       629886,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0])}
```
    
