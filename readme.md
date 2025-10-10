This project implements the code for the paper **"Unsupervised Graph Clustering with Deep Structural Entropy"**, which is accepted to Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2025 (KDD 2025).

If you use this project, please cite:

```bibtex 
@inproceedings{zhang2025unsupervised, 
    title      ={Unsupervised Graph Clustering with Deep Structural Entropy}, 
    author     ={Zhang, Jingyun and Peng, Hao and Sun, Li and Wu, Guanlin and Liu, Chunyang and Yu, Zhengtao}, 
    booktitle  ={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25)},
    year       ={2025}, 
    publisher  ={ACM},
    doi        ={10.1145/3711896.3737173},
    url        ={https://doi.org/10.1145/3711896.3737173}
    } 
```


If you have any questions, feel free to open an issue or email to: BY2339214@buaa.edu.cn


# Environment & Requirements
     -python 3.10.12
     -torch 2.3.1+cu118
     -torch_scatter 2.1.2+pt23cu118
     -torch_geometric 2.5.3
     -numpy 1.26.3
     -dgl 2.3.0+cu118
     -scipy 1.14.0
     -munkres 1.1.4

# Project Structure

`main.py`: Entry point for training or evaluation.

`model.py`: GCN, ASS, and DeSE models.

`utility/`: Utility functions.

└──  `dataset.py`: Data loading and preprocessing.

└──  `parser.py`: Parameters setting and description.

└──  `util.py`: cluster metrics and conversion functions.


# Quick Start

`python DeSE/main.py`
