CUDA_VISIBLE_DEVICES=1  \ 
nohup python get_hedge.py \
    --input_path /home/dataset/Arxiv/pubmed-dataset \
    --output_path dataset/pubmed \
    --task train > get_hedges_train_pubmed.log 2>&1 &