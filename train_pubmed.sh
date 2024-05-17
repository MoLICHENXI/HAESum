CUDA_VISIBLE_DEVICES=1  \
nohup python train_hyper.py \
    --cuda \
    --gpu 1 \
    --n_epochs 13 \
    --data_dir /data/zcl/metahypergraph/dataset/pubmed  \
    --cache_dir /data/zcl/metahypergraph/cache/pubmed \
    --embedding_path /data/zcl/metahypergraph/glove.42B.300d.txt \
    --model Hypergraph \
    --save_root /data/zcl/hyergraph/models/pubmed250  \
    --log_root /data/zcl/hyergraph/log_pubmed \
    --bert_path /data/zcl/metahypergraph/bert_features_pubmed \
    --sent_max_len 100 \
    --lr_descent \
    --grad_clip \
    -m 3  >pubmed_250.log 2>&1 &