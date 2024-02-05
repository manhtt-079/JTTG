DATA_DIR=/mnt/lustre/home/howard/lib_text2text/data/tokenized/train/
OUTPUT_DIR=storages/
MODEL_RECOVER_PATH=/mnt/lustre/home/howard/lib_text2text/storages/unilm1-base-cased.bin
export CUDA_VISIBLE_DEVICES=0,1,2,3
python unilm/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-base-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.pa.tok.txt --tgt_file train.q.tok.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_position_embeddings 512 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 36 --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 1