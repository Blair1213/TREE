# [NBME] Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning
This project is a Transformer-based graph representation learning framework for pan-cancer gene identifications. It applies to both homogeneous networks (PPI network)and heterogeneous networks. TREE can provide generalizable robust and accurate cancer gene identifications with multi-omics-level and network structure-level interpretability. 

![TREE framework](https://github.com/Blair1213/TREE/blob/main/TREE_architecture.jp2)

# Installation & Dependencies
The code is written in Python 3 and was mainly tested on Python 3.6 and a Linux OS. 

TREE has the following dependencies:
* h5py 3.1.0
* keras 2.6.0
* numpy 1.19.5
* networkx 2.5.1
* pandas 1.1.5
* scikit-learn 0.24.2
* tensorflow 2.6.2
* shap 0.41.0


# Datasets
TREE is trained and tested on eight networks, including six homogeneous networks and two heterogeneous networks. The homogeneous networks are composed of gene and gene-gene interactions, while the heterogeneous networks are composed by (gene, miRNA, transcription factor) and (gene, lncRNA, transcription factor).

All pan-cancer networks can be downloaded from [here](https://zenodo.org/records/11648891).

In addition, TREE is also trained and tested on cancer-specific networks, including 16 homogeneous networks and 15 heterogeneous networks. All cancer-specific homogemeous networks are available at https://zenodo.org/records/11648365, and cancer-specific heterogeneous networks are available at https://zenodo.org/records/11648733.

# Reproducibility
To reproduce the results of TREE. Firstly, you are supposed to download the networks and put it into a file "dataset/networks/". Then, you are supposed to download the subgraphs generated for each node from [pdata for pan-cancer networks](https://zenodo.org/records/15045885) and unzip to "pdata/.". In addition, the distance of shortest path for each network can also be downloaded from [shortest path (sp) for pan-cancer networks](https://zenodo.org/records/15045711) and unzip to "sp/.". 

Please note even if you don't download above two files, the code will still run successfully, as these files will be generated during execution.


The directory structure of TREE is:
```
.
├── README.md
├── callbacks
│   ├── __init__.py
│   ├── ensemble.py
│   └── eval.py
├── config.py
├── dataset
│   └── networks
│       ├── CPDB_multiomics.h5
│       ├── ......
├── layers
│   ├── __init__.py
│   ├── attentionFusion.py
│   ├── centralityEncoding.py
│   ├── graphormerEncoder.py
│   ├── multiHeadAttention.py
│   └── spatialEncoding.py
├── losses
│   ├── __init__.py
│   └── weightedBinaryCrossEntropy.py
├── main.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   └── tree.py
├── pdata
│   ├── CPDB_method_rw_channel_6_neighbor_8_spatial.npy
│   ├── CPDB_method_rw_channel_6_neighbor_8_subgraphs.npy
│   ├── ......
├── run.py
├── sp
│   ├── CPDB_sp.h5
│   ├── ......
└── utils
    ├── __init__.py
    ├── data_loader.py
    ├── io.py
    ├── node2vec.py
    └── walker.py

```
Then, you can train TREE by:
```
    python run.py
```
## Citation
```
@article{su2025interpretable,
  title={Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning},
  author={Su, Xiaorui and Hu, Pengwei and Li, Dongxu and Zhao, Bowei and Niu, Zhaomeng and Herget, Thomas and Yu, Philip S and Hu, Lun},
  journal={Nature Biomedical Engineering},
  pages={1--19},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
