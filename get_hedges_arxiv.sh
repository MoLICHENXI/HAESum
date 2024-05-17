CUDA_VISIBLE_DEVICES=1  \ 
nohup python get_hedge.py \
    --input_path /home/dataset/Arxiv/arxiv-dataset \
    --output_path dataset/arxiv \
    --task train > get_hedges_train_arxiv.log 2>&1 &