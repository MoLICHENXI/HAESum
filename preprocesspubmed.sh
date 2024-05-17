CUDA_VISIBLE_DEVICES='0' \ 
nohup python preprocess_data.py \
    --input_path dataset/pubmed-dataset \
    --output_path dataset/pubmed \
    --task train > Preprocess_train_pubmed.log 2>&1 &
     