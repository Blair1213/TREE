# 	TREE
This project is a Transformer-based graph representation learning framework for pan-cancer gene identifications. It applies to both homogeneous networks (PPI network)and heterogeneous networks. TREE can provide generalizable robust and accurate cancer gene identifications with multi-omics-level and network structure-level interpretability. 

# Installation & Dependencies
The code is written in Python 3 and was mainly tested on Python 3.6 and a Linux OS. 

TREE has the following dependencies:



# Datasets
TREE is trained and tested on eight networks, including six homogeneous networks and two heterogeneous networks. The homogeneous networks are composed of gene and gene-gene interactions, while the heterogeneous networks are composed by (gene, miRNA, transcription factor) and (gene, lncRNA, transcription factor).

All networks can be downloaded from http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/network.zip.

If you only focus on homogeneous networks, the link is http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/homogeneous.zip, and http://bioinformatics.tianshanzw.cn:8888/TREE/Networks/heterogeneous.zip for heterogeneous networks.

# Reproducibility
To reproduce the results of TREE. Firstly, you are supposed to download the networks and put it into a file "dataset/networks/". Then, you are supposed to download the subgraphs generated for each node from http://bioinformatics.tianshanzw.cn:8888/TREE/pdata.zip and unzip in the current directory. In addition, the distance of shortest path for each network can also be downloaded from http://bioinformatics.tianshanzw.cn:8888/TREE/sp.zip. The directory structure of TREE is:
、、、
.
├── README.md
├── callbacks
├── config.py
├── dataset
│   └── networks
├── layers
├── log
├── losses
├── main.py
├── models
├── pdata
├── run.py
├── sp
└── utils

    python run.py

