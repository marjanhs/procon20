# Stance Prediction for Contemporary Issues: Data and Experiments


We investigate whether pre-trained bidirectional transformers with sentiment and emotion information improve stance detection in long discussions of contemporary issues. As a part of this work, we create a novel stance detection dataset covering 419 different controversial issues and their related pros and cons collected by procon.org in nonpartisan format. Experimental results show that a shallow recurrent neural network with sentiment or emotion information can reach competitive results compared to fine-tuned BERT with 20x fewer parameters. We also use a simple approach that explains which input phrases contribute to stance detection.




# Paper:
The paper can be found [here](https://www.aclweb.org/anthology/2020.socialnlp-1.5/).

# Citation:
```
@inproceedings{hosseinia-etal-2020-stance,
    title = "Stance Prediction for Contemporary Issues: Data and Experiments",
    author = "Hosseinia, Marjan  and Dragut, Eduard  and Mukherjee, Arjun",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.5",
    pages = "32--40",
    
}
```

# Code: 
The code is an extention of [Hedwig](https://github.com/marjanhs/hedwig) implementation of BERT. Please follow Hedwig's instruction to install the requirements. (you do not need to install word embedding for this project)

# Dataset:

Procon20 contains 419 different controversial issues with 6094 samples. Each sample is a pair of a *(question, argument)* that is either a *pro*(01) or a *con*(10). The dataset file can be found at ```data/ProconDual```. Place dataset files in  ```../hedwig-data/datasets/ProconDual/```


# Model Execution:
 To train and evaluate the VADER-sent-GRU model on (train.tsv, dev.tsv, test.tsv):

```
python -m models.bert_lstm  --dataset ProconDual  --model bert-base-uncased --max-seq-length 256 --batch-size 8 --lr 2e-4 --epochs 30  --gpu 1 --early_on_f1 --seed 2035  --pooling

```

To train and evaluate the NRC-Emotion-GRU model on on (train.tsv, dev.tsv, test.tsv):

```
python -m models.bert_lstm_emotion  --dataset ProconDual  --model bert-base-uncased --max-seq-length 256 --batch-size 8 --lr 2e-4 --epochs 30  --gpu 1 --early_on_f1 --seed 2035  --max-em-len 11 --pooling --emotion-filters positive,negative
```




