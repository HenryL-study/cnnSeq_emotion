# cnnSeq_emotion

##### Description:

A seq2seq model with attention mechanism and a CNN encoder to enhance the ability.

Use for sentence emotion classification.

##### Usage:

`process_questions.py`: Processing sentences in two files. Negative file and positive file. It also need a word vector file to generate word embedding layer data. Here we use `glove.twitter.27B.200d.txt` which can download in [Glove web site](https://nlp.stanford.edu/projects/glove/).

It will generate 6 files for model to use.

`main.py`: Run it for training. Parameters can set in the code.

`word_int_to_word.py`: Translate numbers to words.