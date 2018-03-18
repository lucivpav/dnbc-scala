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
|Number of hidden states         |10   |
|Sequence length                 |200  |
|Observed discrete variables     |5    |
|Observed continuous variables   |5    |
|Learning set length (#sequences)|1000 |
|Testing set length (#sequences) |200  |
|Max Gaussians per mixture       |3    |
|Transitions per hidden state    |5    |

### Machine

|Property |Value    |
|---------|---------|
|Processor|i5-7200U |
|Memory   |8GB      |
|System   |Fedora 27|

### Results

|Property                |Workers=1|Workers=2|
|------------------------|----------|--------|
|Learning time [s]       |140       |85      |
|Testing time [s]        |4         |4       |

Average success rate: 83%
