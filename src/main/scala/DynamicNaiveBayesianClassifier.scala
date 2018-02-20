import org.apache.spark.SparkContext

import scala.collection.mutable.ListBuffer

// Usage: call learnSequence() once per each data in training set
//        then finalize learning stage by calling learnFinalize()
// continuousVariableHints - assumed number of normal distributions in a
//                            particular continuous variable (Gaussian mixture), default is 1
class DynamicNaiveBayesianClassifier(sc: SparkContext,
                                      sequenceLength: Int,
                                      discreteVariablesCount: Int, continuousVariablesCount: Int,
                                      var continuousVariableHints: List[Int] = List.empty) {
  private var initialEdge: DiscreteEdge = new DiscreteEdge
  private var transitions: Map[String, DiscreteEdge] = Map.empty
  private var discreteEmissions: Array[Map[String, DiscreteEdge]] = Array.ofDim(discreteVariablesCount)
  private var continuousEmissions: Array[Map[String, ContinuousEdge]] = Array.ofDim(continuousVariablesCount)

  initializeEmissions()
  checkValidHints()
  initializeVariableHints()

  def learnSequence(hiddenStates: List[String],
                    observedDiscreteVariables: List[List[String]],
                    observedContinuousVariables: List[List[Double]]): Unit = {
    checkValidHiddenStates(hiddenStates)
    checkValidObservations(observedDiscreteVariables, observedContinuousVariables)
    learnDiscreteEmissions(hiddenStates, observedDiscreteVariables)
    learnContinuousEmissions(hiddenStates, observedContinuousVariables)
    learnTransitions(hiddenStates)
    learnInitialEdge(hiddenStates)
  }

  def learnFinalize(): Unit = {
    initialEdge.learnFinalize()
    transitions.foreach(t => t._2.learnFinalize())
    discreteEmissions.foreach(e => e.foreach(e2 => e2._2.learnFinalize()))
    continuousEmissions.foreach(e => e.foreach(e2 => e2._2.learnFinalize()))
  }

  def infereMostLikelyHiddenStates(observedDiscreteVariables: List[List[String]],
                                    observedContinuousVariables: List[List[Double]]): List[String] = {
    checkValidObservations(observedDiscreteVariables, observedContinuousVariables)
    var probs = viterbiInitialize(observedDiscreteVariables, observedContinuousVariables)
    viterbiCompute(probs, observedDiscreteVariables, observedContinuousVariables)
  }

  private def initializeEmissions(): Unit = {
    for (i <- 0 until discreteVariablesCount)
      discreteEmissions(i) = Map.empty
    for (i <- 0 until continuousVariablesCount)
      continuousEmissions(i) = Map.empty
  }

  private def checkValidHints(): Unit = {
    if ( continuousVariableHints.isEmpty )
      return
    if ( continuousVariableHints.lengthCompare(continuousVariablesCount) != 0 )
      throw new Exception("Number of continuous variable hints does not correspond to the number of continous variables")
    if ( continuousVariableHints.count(h => h < 1 ) != 0 )
      throw new Exception("Number of suggested normal distributions in a continuous variable must be at least one")
  }

  private def initializeVariableHints(): Unit = {
     if (continuousVariableHints.isEmpty) {
      var hints = ListBuffer.empty[Int]
      for ( i <- 0 until continuousVariablesCount )
        hints += 1
      continuousVariableHints = hints.toList
    }
  }

  private def checkValidHiddenStates(hiddenStates: List[String]): Unit = {
    if (hiddenStates.lengthCompare(sequenceLength) != 0)
      throw new Exception("Hidden sequence is longer than expected sequence length")
  }

  private def checkValidObservations(observedDiscreteVariables: List[List[String]],
                                      observedContinuousVariables: List[List[Double]]): Unit = {
    if (observedDiscreteVariables.lengthCompare(discreteVariablesCount) != 0)
      throw new Exception("Number of observed discrete variables does not correspond to count set")
    if (observedContinuousVariables.lengthCompare(continuousVariablesCount) != 0)
      throw new Exception("Number of observed continuous variables does not correspond to count set")

    val lengths = observedDiscreteVariables.map(v => v.length).union(observedContinuousVariables.map(v => v.length))
    if (lengths.count(l => l != sequenceLength) != 0)
      throw new Exception("Sequence to be learned is longer than expected sequence length")
  }

  private def learnInitialEdge(hiddenStates: List[String]): Unit = {
    initialEdge.learn(hiddenStates.head)
  }

  private def learnTransitions(hiddenStates: List[String]): Unit = {
    for (i <- 0 to sequenceLength - 2) {
      val from = hiddenStates(i)
      val to = hiddenStates(i + 1)
      if (!transitions.contains(from))
        transitions += (from -> new DiscreteEdge)
      transitions(from).learn(to)
    }
  }

  private def learnContinuousEmissions(hiddenStates: List[String], observedContinuousVariables: List[List[Double]]): Unit = {
    for (i <- observedContinuousVariables.indices) {
      val states = observedContinuousVariables(i)
      var emission = continuousEmissions(i)
      for (j <- 0 until sequenceLength) {
        val hiddenState = hiddenStates(j)
        val observedState = states(j)
        if (!emission.contains(hiddenState)) {
          val k = continuousVariableHints(i)
          var continuousEdge: ContinuousEdge = null
          if (k == 1)
            continuousEdge = new ContinuousGaussianEdge
          else
            continuousEdge = new ContinuousGaussianMixtureEdge(sc, continuousVariableHints(i))
          emission += (hiddenState -> continuousEdge)
        }
        emission(hiddenState).learn(observedState)
      }
      continuousEmissions(i) = emission
    }
  }

  private def learnDiscreteEmissions(hiddenStates: List[String], observedDiscreteVariables: List[List[String]]): Unit = {
    for (i <- observedDiscreteVariables.indices) {
      val states = observedDiscreteVariables(i)
      var emission = discreteEmissions(i)
      for (j <- 0 until sequenceLength) {
        val hiddenState = hiddenStates(j)
        val observedState = states(j)
        if (!emission.contains(hiddenState))
          emission += (hiddenState -> new DiscreteEdge)
        emission(hiddenState).learn(observedState)
      }
      discreteEmissions(i) = emission
    }
  }

  private def viterbiInitialize(observedDiscreteVariables: List[List[String]],
                                observedContinuousVariables: List[List[Double]]): Map[String,Double] = {
    var vcur = Map.empty[String,Double]
    for (hiddenState <- transitions.keys) {
      var emissionsSum = 0.0
      emissionsSum += discreteEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observedDiscreteVariables(z._2).head)).sum
      emissionsSum += continuousEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observedContinuousVariables(z._2).head)).sum
      vcur += (hiddenState -> (Math.log(emissionsSum) + Math.log(initialEdge.probability(hiddenState))))
    }
    vcur
  }

  private def viterbiCompute(probs: Map[String,Double],
                             observedDiscreteVariables: List[List[String]],
                             observedContinuousVariables: List[List[Double]]): List[String] = {
    var vcur = Map.empty[String,Double]
    var vprev = probs
    var path = ListBuffer.empty[String]
    for( i <- 1 until sequenceLength )
    {
      path += vprev.maxBy(_._2)._1 // append path
      vcur = Map.empty[String,Double]
      for ( hiddenState <- transitions.keys )
      {
        var emissionsSum = 0.0
        emissionsSum += discreteEmissions.zipWithIndex.map(z => z._1(hiddenState)
                                                            .probability(observedDiscreteVariables(z._2)(i))).sum
        emissionsSum += continuousEmissions.zipWithIndex.map(z => z._1(hiddenState)
                                                              .probability(observedContinuousVariables(z._2)(i))).sum
        vcur += (hiddenState -> (Math.log(emissionsSum) +
                                  vprev.maxBy{case (hs,p) => p + Math.log(transitions(hs).probability(hiddenState))}._2))
      }
      vprev = vcur
    }
    path += vprev.maxBy(_._2)._1
    path.toList
  }
}
