# Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Level
[NAACL 2024 Findings] Source code for our paper 'Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Level'

paper: https://arxiv.org/abs/2405.10202
## Overview
![Image text](https://github.com/MoLICHENXI/Picture/blob/main/figure1.png)
An illustration of modeling an input document from local and global perspectives. Triangles and circles represent words and sentences in the original document respectively.
## Setup
### Installation
The code is written in Python 3.6+. Its dependencies are summarized in the file requirements.txt. You can install these dependencies like this:
```shell
pip install -r requirements.txt
```
### Datasets
Download Pubmed and Arxiv datasets from [here](https://github.com/armancohan/long-summarization).

### Preprocess data
#### For PubMed dataset:
```shell
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task train
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task val
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task test
```
Alternatively, you can directly run **preprocesspubmed.sh**:
```shell
bash preprocesspubmed.sh
```

#### For ArXiv dataset:
```shell
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task train
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task val
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task test
```
Alternatively, you can directly run **preprocessarxiv.sh**:
```shell
bash preprocessarxiv.sh
```


#### Create Vocabulary & TFIDF Vectors
After getting the standard JSON format, you process the **PubMed or Arxiv** dataset by running a script: 
```shell
bash PrepareDataset.sh
```
The processed files will be put under the cache directory.

#### Prepare for hyperedges
##### For PubMed dataset:
```shell
python get_hedge.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task train
python get_hedge.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task val
python get_hedge.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task test
```
or
```shell
bash get_hedges_pubmed.sh
```

##### For ArXiv dataset:
```shell
python get_hedge.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task train
python get_hedge.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task val
python get_hedge.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task test
```
or
```shell
bash get_hedges_arxiv.sh
```
### Training
Run command like this
```shell
python train_hyper.py --cuda --gpu 0 --n_epochs 10 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --model Hypergraph --save_root <model path> --log_root <log path> --lr_descent --grad_clip -m 3
```
Alternatively, you can directly run the corresponding **dataset's .sh** file. For example
```shell
bash tran_arxiv.sh
```
The training time for one epoch on the PubMed dataset on A6000 48G is 3 hours, while on the Arxiv dataset, it is 6 hours.
### Evaluation
For evaluation, the command may like this:
```shell
python evaluation.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path>  --model Hypergraph --save_root <model path> --log_root <log path> -m 5 --test_model multi --use_pyrouge
```
**Note**: To use ROUGE evaluation, you need to download the 'ROUGE-1.5.5' package and then use pyrouge.

**Error Handling**:  If you encounter the error message Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling). Or you can also refer to the [**installation tutorial**](https://github.com/MoLICHENXI/Rouge-installation-guide) provided by the author in Chinese.

## Citation
```bibtex
@article{zhao2024hierarchical,
  title={Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Level},
  author={Zhao, Chenlong and Zhou, Xiwen and Xie, Xiaopeng and Zhang, Yong},
  journal={arXiv preprint arXiv:2405.10202},
  year={2024}
}
```
