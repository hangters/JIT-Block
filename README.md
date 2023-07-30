## Dataset

Unzip JIT-Block-Defect4J and JIT-Defect4J to data/jitblock (Note that the JIT-Defect4J dataset we use is slightly different from the original dataset see the article for details)

### **JIT-Block Implementation**

All of our experiments were performed on an NVIDIA A40, and it is recommended to use the same graphics card or reduce the train_batch_size
To train JIT-Block, run the following command(you can change --max_changed_block_unit to replicate our experiment):

```shell
python -m JITBlock.run \
    --output_dir=model/jitblock/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitblock/changed_block_train.pkl data/jitblock/changed_block_features_train.pkl \
    --eval_data_file data/jitblock/changed_block_valid.pkl data/jitblock/changed_block_features_valid.pkl\
    --test_data_file data/jitblock/changed_block_test.pkl data/jitblock/changed_block_features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 12 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 2 \
    --max_changed_block_unit 5 \
    --seed 42 2>&1| tee model/jitblock/train.log

```

To obtain the evaluation, run the following command:

```shell
python -m JITBlock.run \
    --output_dir=model/jitfine/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/jitblock/changed_block_train.pkl data/jitblock/changed_block_features_train.pkl \
    --eval_data_file data/jitblock/changed_block_valid.pkl data/jitblock/changed_block_features_valid.pkl\
    --test_data_file data/jitblock/changed_block_test.pkl data/jitblock/changed_block_features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 12 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/jitblock/changes_complete_buggy_line_level.pkl \
    --max_changed_block_unit 5 \
    --seed 42 2>&1 | tee model/jitblock/test.log

```

### Ablation Experiment

To do the ablation experiment, run the following command:

```shell
python -m JITBlock.run \
    --output_dir=model/jitblock/checkpoints\
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitblock/changes_train.pkl data/jitblock/features_train.pkl \
    --eval_data_file data/jitblock/changes_valid.pkl data/jitblock/features_valid.pkl\
    --test_data_file data/jitblock/changes_test.pkl data/jitblock/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 12 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --patience 2 \
    --max_changed_block_unit 5 \
    --do_ablation True \
    --seed 42 2>&1| tee model/jitblock/train_ablation.log
```

To obtain the evaluation, run the following command:

```shell
python -m JITBlock.run \
    --output_dir=model/jitblock/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
   --train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl \
    --eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl\
    --test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --max_changed_block_unit 5 \
    --do_ablation True \
    --seed 42 2>&1 | tee model/jitblock/test_ablation.log
```

