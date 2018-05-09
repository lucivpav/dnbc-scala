# dnbc-scala
Parallel implementation of dynamic naive Bayesian classifier
## Accuracy 
Data sets based on [Toy Robot data set](https://www.cs.princeton.edu/courses/archive/fall06/cos402/hw/hw5/hw5.html)

|Data set type                   |Average success rate [%]|
|--------------------------------|------------------------|
|Discrete                        |65                      |
|Continuous                      |42                      |
|Bivariate                       |76                      |
|Gaussian mixture (without hint) |96                      |
|Gaussian mixture (with hint)    |99                      |

The average success rate means the average percentage of hidden states inferred correctly.

There are two main reasons for relatively low overall sucess rate:

1) Only about 90% of observed symbols are accurate
2) There are multiple transitions to hidden states with the same observed symbol

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

|Property |Value                                  |
|---------|---------------------------------------|
|Processor|2× 8-core Intel Xeon E5-2650 v2 2.6 GHz|
|Memory   |15 GB                                  |
|Disk     |10 GB HDD                              |

### Results

|Property                |Workers=1|Workers=2|Workers=4|Workers=8|Workers=15|
|------------------------|---------|---------|---------|---------|----------|
|Learning time speed up  |1        |1.3      |1.5      |1.8      |2         |
