# simple-hmm
Implementation of Hidden Markov Models with discrete states
## Performance 
Measurements were made on [Toy Robot dataset](https://www.cs.princeton.edu/courses/archive/fall06/cos402/hw/hw5/hw5.html)

|Method|Average success rate [%]|
|--------|------------------------|
|Random  |0.17                    |
|HMM     |1.4                     |

The average success rate means the number of hidden states inferred correctly until an incorrect state.
Since the most challenging part part of the HMM inference algorithm is to predict the initial hidden state,
I only consider cases where the initial state was inferred correctly.

As a demonstration that the implementation produces reasonable results, it was compared to a random method,
that selects hidden states at random. As seen in the table, my implementation of HMM is **7x better** than the random method.

There are two main reasons for relatively low overall sucess rate of HMM method:

1) 10% of observed states don't correspond to ground truth
2) There are more than one similarly likely transitions from a particular hidden state
