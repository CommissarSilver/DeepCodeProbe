Attention-based Tree-to-Sequence Code Summarization Model
The TensorFlow Eager Execution implementation of Source Code Summarization with Extended Tree-LSTM (Shido+, 2019)

including:

Multi-way Tree-LSTM model (Ours)
Child-sum Tree-LSTM model
N-ary Tree-LSTM model
DeepCom (Hu et al.)
CODE-NN (Iyer et al.)
Dataset
Download raw dataset from [https://github.com/xing-hu/DeepCom]
Parse them with parser.jar
Usage
Prepare tree-structured data with dataset.py
Run $ python dataset.py [dir]
Train and evaluate model with train.py
See $ python train.py -h