CUDA_VISIBLE_DEVICES='0' \ 
nohup python preprocess_data.py \
    --input_path dataset/arxiv-dataset \
    --output_path dataset/arxiv \
    --task train > Preprocess__train_arxiv.log 2>&1 &