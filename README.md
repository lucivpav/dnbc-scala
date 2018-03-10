# dnbc-scala
Parallel implementation of dynamic naive Bayesian classifier
## Accuracy 
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

## Performance
### Data set

|Property                        |Value|
|--------------------------------|-----|
|Number of hidden states         |36   |
|Sequence length                 |200  |
|Observed variables              |40   |
|Learning set length (#sequences)|200  |
|Testing set length (#sequences) |200  |

### Machine

|Property |Value    |
|---------|---------|
|Processor|i5-7200U |
|Memory   |8GB      |
|System   |Fedora 27|

### Results

|Property                |Sequential|Parallel|
|------------------------|----------|--------|
|Average success rate [%]|18        |17      |
|Learning time [s]       |40        |30      |
|Testing time [s]        |63        |58*     |

<sub>\* Which is a bit strange, considering inference stage has not been parallelized.</sub>
