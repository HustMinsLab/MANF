# MANF
This repository provides the code of MANF model for the Findings of ACL 2022 paper: Encoding and Fusing Semantic Connection and Linguistic Evidence for Implicit Discourse Relation Recognition.

# Data
We use the PDTB 3.0 corpus for evaluation. Due to the LDC policy, we cannot release the PDTB data. If you have bought data from LDC, please put the PDTB .tsv file in dataset.

# Requirements
python 3.8.8  
torch == 1.8.0

# How to use
- You have to put the PDTB corpus tsv file in dataset file first.
- Then run it:  
```
python main_BERT.py
python main_BiLSTM.py
```

# Citation
Please cite our paper if you use the code!
