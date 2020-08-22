<h1>CNN</h1>

Link to source in picture bellow:

[![Everything Is AWESOME](./imgs/rnn.jpg)](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

<!-- TOC -->

- [For which types of data do we use RNNs?](#for-which-types-of-data-do-we-use-rnns)
- [What is special with the training of RNNs?](#what-is-special-with-the-training-of-rnns)
- [What is an LSTM?](#what-is-an-lstm)
- [Deepdive: Find out which parameters are trainable in an LSTM](#deepdive-find-out-which-parameters-are-trainable-in-an-lstm)

<!-- /TOC -->



# For which types of data do we use RNNs?

We use RNNs in context of data where we have a sequence variable. this can be in context of
1) the order of words in a sentence | NLP
   1) Language Modelling & Generating text content
   2) Machine translaotion e.g translation ebtween German and English
   3) Speech recognition
   4) Generating Image Descriptions (RNN in connection with a [CNN](./05_CNNs.md))
2) data in context of time => time series data


In "normal"(=no RNN) neuonal networks the input data has no real interconnection and can be processed indivualy. In RNN we have a kind of Stream of data. Consequently we have following parameters and

According the title image:

X = input data
s = current state
t = time

U, V and W are the parameters of the network that you need to learn from your training data. So before a network is trained we don't usually know the "content" of these matrices. These Parameters are shared over time


V = how to map the hidden state (memory) back into the space of possible
U = kind of input weight
W = Value of the state given to the next layer

Together U and W define how to calculate the new memory of the network given the previous memory.
V defines how to map the hidden state (memory) back into the space of possible


# What is special with the training of RNNs?

Like the other ["normal" Neuroal Networks](./03_Neural_Networks.md) we use the Backpropagation for the training. In each timestamp we  we execute the BPO. In comparision to the normal NNs we update the common values U, V and W. These updates on a specific timestamp on these parameters have an impact to the predcition/training to the preivious and following operations. Consequently we cant update the values solo specific for the current timestamp in focus. If we update the parameters at timenstamp n^t, we have have to take in consideration all steps with < n^t. This process is calle as Backpropagation Through Time (BPTT) .

# What is an LSTM?
A LongShortTermMemory Neuronal Network use a different function to compute the hidden state. In comparison to the other RNNs cells LSTMs cell decide if they use the W which is transfered from the previous cell or if they delete the preivious W. With this ability it is possible to buil non "theoretical endless" LSTM Chains. The Calculation of the current state is consequently 
- the decision of using the previous information
- the current input data
- the current memory

This type of RNNs is efficient especially with long term history / Sequence data


# Deepdive: Find out which parameters are trainable in an LSTM

According to this [Tutorial](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)

we have following Hyperparameter in a LSTM:

1) Number of Epochs
2) Batch Size
3) Number of Neurons
4) 




