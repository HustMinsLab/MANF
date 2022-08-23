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

@inproceedings{Wei.X:et.al:2022:ACL-Fingdings,  
>>title = "Encoding and Fusing Semantic Connection and Linguistic Evidence for Implicit Discourse Relation Recognition",  
    author = "Xiang, Wei and Wang, Bang and Dai, Lu and Mo, Yijun",  
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",  
    month = may,  
    year = "2022",  
    address = "Dublin, Ireland",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2022.findings-acl.256",  
    doi = "10.18653/v1/2022.findings-acl.256",  
    pages = "3247--3257",  
}
