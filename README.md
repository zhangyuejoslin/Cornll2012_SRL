# CSE 891: Semantic Role Labeling Group Project

## Dataset
Semantic role labeling extracts predicate argument relations, such as “who did what to whom.”  In this group project, we use CONLL2012 dataset, where the data find in this link: http://cemantix.org/data/ontonotes.html. To better cope with this data, we have a data preprocessing, and the data stored in this link: https://github.com/zhangyuejoslin/Cornll2012_SRL/tree/master/data/BIO-formatted

## Data example
Here it is the data example:

> 1 To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast . ||| O B-V B-ARG1 I-ARG1 O B-ARG0 I-ARG0 I-ARG0 I-ARG0 I-ARG0 O O O O O O O O O O O O

> 10 To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast . ||| B-R-ARG0 I-R-ARG0 I-R-ARG0 I-R-ARG0 O B-ARG0 I-ARG0 I-ARG0 I-ARG0 I-ARG0 B-V B-ARG1 I-ARG1 I-ARG1 B-ARG2 I-ARG2 I-ARG2 I-ARG2 I-ARG2 I-ARG2 I-ARG2 O

> 18 To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast . ||| O O O O O O O O O O O O O O O O O O B-V O O O

> 19 To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast . ||| O O O O O O O O O O O O O O O O O O O B-V O O

> 20 To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast . ||| O O O O O O O O O O O O O O O B-ARG1 I-ARG1 B-R-ARG1 O O B-V O

Since the sample may has more than one verb, each sentence of this data is annotated for all verbs by BIO-Tagging. eg. In the first sample:

1. "1" is the verb location index, 
2. "To express its determination , the Chinese securities regulatory department compares this stock reform to a die that has been cast ." is the raw set;
3. ||| is the signal to split between raw sentence and BIO-tagging;
4. O B-V B-ARG1 I-ARG1 O B-ARG0 I-ARG0 I-ARG0 I-ARG0 I-ARG0 O O O O O O O O O O O O is the BIO-tagging.

## Data Statstic
1. Traing data size: 240992
2. Dev data size: 35297
3. Test data size: 26715
4. Word size: 400001

## Data Reader

In the Data Reader step, we replace word to the word index, in order to service with the model part. Furthermore, we load the Glove pre-training embedding and the embedding vocabulary is same as the dataset bocabulary. We set the max length of each data sample to 20, and we build a mask to each data.

After data reader, the data is like this format:

1. first train sample data: [  66 5204    2   21   16 7994  738 1220   22    1  918 4909  936    3  0    0    0    0    0    0]

2. first train sample mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]

3. first train sample label: [12, 12, 12, 12, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12]

4. first train sample predicate position: 4

----------------------------------------------------

1. first dev sample data: [   7    1  741    4  710    2    8 1836   13   70   34  174   52  863
  991    5  552 3041   18 6399]
2. first dev sample mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
3. first dev sample label: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
4. first dev sample predicate position: 10
----------------------------------------------------

## Model Description and Current Performance
1. Bi-LSTM as baseline
>-  We implement this baseline by a two-layer bidirectional LSTM with 100 hidden units. 
>- We got 55.694% and 58.174% micro-averaged multiclass F1 score for develop and test dataset respectively.
2. Bi-LSTM+Highway+Viterbi Model
>- We implement this model with three different modules. 
>- The first module is the bidirectional LSTM. To better compare the performance, we maintain the same hyper-parameters, which are the same as the baseline. 
>- The second module is a highway connection module that can alleviate the vanishing gradient problem when training deep BiLSTMs. 
>- The third module is the Viterbi algorithm. The approach described so far does not model any dependencies between the output tags. To incorporate constraints on the output structure at decoding time, we make use of the Viterbi algorithm for decoding. All the constraints are based on the original paper.
>- Finally, we obtain 57.00% and 59.49% micro-averaged multiclass F1 score for dev and test dataset.





