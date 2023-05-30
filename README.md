# 	TREE
This project is a Transformer-based graph representation learning framework for pan-cancer gene identifications. It applies to both homogeneous networks (PPI network)and heterogeneous networks. TREE can provide generalizable robust and accurate cancer gene identifications with multi-omics-level and network structure-level interpretability. 

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

All networks can be downloaded from http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/network.zip.

If you only focus on homogeneous networks, the link is http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/homogeneous.zip, and http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/heterogeneous.zip for heterogeneous networks.

# Reproducibility
To reproduce the results of TREE. Firstly, you are supposed to download the networks and put it into a file "dataset/networks/". Then, you are supposed to download the subgraphs generated for each node from http://bioinformatics.tianshanzw.cn:8888/TREE/pdata.zip and unzip in the current directory. In addition, the distance of shortest path for each network can also be downloaded from http://bioinformatics.tianshanzw.cn:8888/TREE/sp.zip. The directory structure of TREE is:
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

