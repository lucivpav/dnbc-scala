# dnbc-scala
Parallel implementation of dynamic naive Bayesian classifier
## Performance 
Data sets based on [Toy Robot data set](https://www.cs.princeton.edu/courses/archive/fall06/cos402/hw/hw5/hw5.html)

|Data set type|Average success rate [%]|
|-------------|------------------------|
|Discrete     |65                      |
|Continuous   |42                      |
|Bivariate    |60                      |

The average success rate means the average percentage of hidden states inferred correctly.

There are two main reasons for relatively low overall sucess rate:

1) Only about 90% of observed states are accurate
2) There are more than one similarly likely transitions from a particular hidden state
