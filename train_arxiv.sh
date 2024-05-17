CUDA_VISIBLE_DEVICES=2  \
nohup python train_hyper.py \
    --cuda \
    --gpu 2 \
    --data_dir /home/code/diffusion/MTGNN/dataset/arxiv  \
    --cache_dir /home/code/diffusion/MTGNN/cache/arxiv \
    --embedding_path /home/code/diffusion/MTGNN/glove.42B.300d.txt \
    --model Hypergraph \
    --save_root /home/code/diffusion/metahypergraph/models_arxiv/arxiv200 \
    --log_root /home/code/diffusion/metahypergraph/log_arxiv \
    --bert_path /home/code/diffusion/MTGNN/bert_features_arxiv \
    --lr_descent \
    --grad_clip \
    -m 3  >arxiv_200withoutglobal_new.log 2>&1 &