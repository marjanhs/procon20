# Stance Prediction for Contemporary Issues: Data and Experiments


We investigate whether pre-trained bidirectional transformers with sentiment and emotion information improve stance detection in long discussions of contemporary issues. As a part of this work, we create a novel stance detection dataset covering 419 different controversial issues and their related pros and cons collected by procon.org in nonpartisan format. Experimental results show that a shallow recurrent neural network with sentiment or emotion information can reach competitive results compared to fine-tuned BERT with 20x fewer parameters. We also use a simple approach that explains which input phrases contribute to stance detection.


# Paper:
The paper can be found [here](https://www.aclweb.org/anthology/2020.socialnlp-1.5/).

# Dataset:

Procon20 contains 419 different controversial issues with 6094 samples. Each sample is a pair of a *(question, argument)* that is either a *pro*(01) or a *con*(10). The dataset file can be found here at root. 

