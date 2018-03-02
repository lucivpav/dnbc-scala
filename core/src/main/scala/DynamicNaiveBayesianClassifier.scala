import org.apache.spark.SparkContext

import scala.collection.mutable.ListBuffer

/**
  * Learned model used for inference
  * @param initialEdge initial probabilities
  * @param transitions discrete probabilities
  * @param discreteEmissions number of items in the list corresponds to the number of discrete observed variables
  * @param continuousEmissions number of items in the list corresponds to the number of continuous observed variables
  */
class DynamicNaiveBayesianClassifier(initialEdge: DiscreteEdge,
                                     transitions: Map[String, DiscreteEdge],
                                     discreteEmissions: List[Map[String, DiscreteEdge]],
                                     continuousEmissions: List[Map[String, ContinuousEdge]]) {

  /**
    * Infers the most likely sequence of hidden states given observations
    * @param observedStates number of observed variables must be the same as provided in the constructor
    * @return a sequence of most likely hidden states
    */
  def inferMostLikelyHiddenStates(observedStates: Seq[ObservedState]): List[String] = {
    checkValidObservations(observedStates)
    val probs = viterbiInitialize(observedStates)
    viterbiCompute(probs, observedStates)
  }

  private def checkValidObservations(states: Seq[ObservedState]): Unit = {
    val discreteVariablesCount = discreteEmissions.length
    val continuousVariablesCount = continuousEmissions.length
    if ( states.exists( s => s.DiscreteVariables.lengthCompare(discreteVariablesCount) != 0 ||
                              s.ContinuousVariables.lengthCompare(continuousVariablesCount) != 0 ) )
      throw new Exception("Number of observed variables is inconsistent")
  }

  private def viterbiInitialize(observedStates: Seq[ObservedState]): Map[String,Double] = {
    var vcur = Map.empty[String,Double]
    for (hiddenState <- transitions.keys) {
      var emissionsSum = 0.0
      emissionsSum += discreteEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observedStates.head.DiscreteVariables(z._2))).sum
      emissionsSum += continuousEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observedStates.head.ContinuousVariables(z._2))).sum
      vcur += (hiddenState -> (Math.log(emissionsSum) + Math.log(initialEdge.probability(hiddenState))))
    }
    vcur
  }

  private def viterbiCompute(probs: Map[String,Double],
                             observedStates: Seq[ObservedState]): List[String] = {
    val sequenceLength = observedStates.length
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
                                                            .probability(observedStates(i).DiscreteVariables(z._2))).sum
        emissionsSum += continuousEmissions.zipWithIndex.map(z => z._1(hiddenState)
                                                              .probability(observedStates(i).ContinuousVariables(z._2))).sum
        vcur += (hiddenState -> (Math.log(emissionsSum) +
                                  vprev.maxBy{case (hs,p) => p + Math.log(transitions(hs).probability(hiddenState))}._2))
      }
      vprev = vcur
    }
    path += vprev.maxBy(_._2)._1
    path.toList
  }
}

/**
  * All observed variables at given time point
  * @param discreteVariables the states of discrete variables
  * @param continuousVariables the states of continuous variables
  */
class ObservedState(discreteVariables: List[String], continuousVariables: List[Double]) {
  def DiscreteVariables: List[String] = discreteVariables
  def ContinuousVariables: List[Double] = continuousVariables
}

/**
  * Observed and hidden state at given time point
  * @param hiddenState
  * @param observedState
  */
class State(hiddenState: String, observedState: ObservedState) {
  def HiddenState: String = hiddenState
  def ObservedState: ObservedState = observedState
}

/** Factory for DynamicNaiveBayesianClassifier instances */
object DynamicNaiveBayesianClassifier {
  /**
    * Creates a DynamicNaiveBayesianClassifier model learned from sequences using maximum likelihood estimation
    * @param sc spark context
    * @param sequences sequences to be learned from
    * @param continuousVariableHints assumed number of normal distributions in a particular continuous variable
    *                                (GaussinMixture), default is 1 per each continuous variable
    * @return learned DynamicNaiveBayesicalClassifier model
    */
  def mle(sc: SparkContext,
          sequences: Iterable[Seq[State]],
          continuousVariableHints: Option[List[Int]] = Option.empty)
          : DynamicNaiveBayesianClassifier = {

    val firstSequence = sequences.take(1)
    val firstDataPoint = firstSequence.toList.head.take(1)
    val discreteVariablesCount = firstDataPoint.head.ObservedState.DiscreteVariables.length
    val continuousVariablesCount = firstDataPoint.head.ObservedState.ContinuousVariables.length
    val originalSequences = firstSequence ++ sequences // TODO: how is ++ on iterables implemented? Hopefully not .toList

    val discreteEmissions: Array[Map[String, DiscreteEdge]] = Array.ofDim(discreteVariablesCount)
    val continuousEmissions: Array[Map[String, ContinuousEdge]] = Array.ofDim(continuousVariablesCount)
    var transitions: Map[String, DiscreteEdge] = Map.empty
    val initialEdge: DiscreteEdge = new DiscreteEdge
    val hints = continuousVariableHints.getOrElse(List.empty)

    initializeEmissions(discreteVariablesCount, continuousVariablesCount, discreteEmissions, continuousEmissions)
    checkValidHints(hints, continuousVariablesCount)
    val initializedHints = initializeVariableHints(hints, continuousVariablesCount)

    originalSequences.foreach { seq =>
      if ( seq.exists( dp => dp.ObservedState.DiscreteVariables.lengthCompare(discreteVariablesCount) != 0 ||
                              dp.ObservedState.ContinuousVariables.lengthCompare(continuousVariablesCount) != 0 ) )
        throw new Exception("Number of observed variables is inconsistent")
      val hiddenStates = seq.map(dp => dp.HiddenState).toList
      val observedDiscreteVariables = (0 until discreteVariablesCount).map(i => seq.map(
                                        s => s.ObservedState.DiscreteVariables(i)).toList).toList
      val observedContinuousVariables = (0 until continuousVariablesCount).map(i => seq.map(
                                        s => s.ObservedState.ContinuousVariables(i)).toList).toList
      learnDiscreteEmissions(discreteEmissions, hiddenStates, observedDiscreteVariables)
      learnContinuousEmissions(sc, continuousEmissions, initializedHints, hiddenStates,
                                observedContinuousVariables)
      transitions = learnTransitions(transitions, hiddenStates)
      learnInitialEdge(initialEdge, hiddenStates)
    }
    learnFinalize(initialEdge, transitions, discreteEmissions, continuousEmissions)
    new DynamicNaiveBayesianClassifier(initialEdge, transitions, discreteEmissions.toList, continuousEmissions.toList)
  }

  private def learnDiscreteEmissions(discreteEmissions: Array[Map[String, DiscreteEdge]],
                                     hiddenStates: List[String], observedDiscreteVariables: List[List[String]]): Unit = {
    val sequenceLength = hiddenStates.length
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

  private def learnContinuousEmissions(sc: SparkContext,
                                       continuousEmissions: Array[Map[String, ContinuousEdge]],
                                       continuousVariableHints: List[Int],
                                       hiddenStates: List[String], observedContinuousVariables: List[List[Double]])
                                        : Unit = {
    val sequenceLength = hiddenStates.length
    for (i <- observedContinuousVariables.indices) {
      val states = observedContinuousVariables(i)
      var emission = continuousEmissions(i)
      for (j <- 0 until sequenceLength) {
        val hiddenState = hiddenStates(j)
        val observedState = states(j)
        if (!emission.contains(hiddenState)) {
          val k = continuousVariableHints(i)
          var continuousEdge: ContinuousEdge = null
          continuousEdge = new ContinuousEdge(sc, continuousVariableHints(i))
          emission += (hiddenState -> continuousEdge)
        }
        emission(hiddenState).learn(observedState)
      }
      continuousEmissions(i) = emission
    }
  }

  private def learnTransitions(transitions: Map[String, DiscreteEdge],
                               hiddenStates: List[String]): Map[String, DiscreteEdge] = {
    val sequenceLength = hiddenStates.length
    var updatedTransitions = transitions
    for (i <- 0 to sequenceLength - 2) {
      val from = hiddenStates(i)
      val to = hiddenStates(i + 1)
      if (!transitions.contains(from))
        updatedTransitions += (from -> new DiscreteEdge)
      updatedTransitions(from).learn(to)
    }
    updatedTransitions
  }

  private def learnInitialEdge(initialEdge: DiscreteEdge, hiddenStates: List[String]): Unit = {
    initialEdge.learn(hiddenStates.head)
  }

  private def learnFinalize(initialEdge: DiscreteEdge, transitions: Map[String, DiscreteEdge],
                            discreteEmissions: Array[Map[String, DiscreteEdge]],
                            continuousEmissions: Array[Map[String, ContinuousEdge]]): Unit = {
    initialEdge.learnFinalize()
    transitions.foreach(t => t._2.learnFinalize())
    discreteEmissions.foreach(e => e.foreach(e2 => e2._2.learnFinalize()))
    continuousEmissions.foreach(e => e.foreach(e2 => e2._2.learnFinalize()))
  }

  private def initializeEmissions(discreteVariablesCount: Int, continuousVariablesCount: Int,
                                  discreteEmissions: Array[Map[String, DiscreteEdge]],
                                  continuousEmissions: Array[Map[String, ContinuousEdge]]): Unit = {
    for (i <- 0 until discreteVariablesCount)
      discreteEmissions(i) = Map.empty
    for (i <- 0 until continuousVariablesCount)
      continuousEmissions(i) = Map.empty
  }

  private def checkValidHints(continuousVariableHints: List[Int], continuousVariablesCount: Int): Unit = {
    if ( continuousVariableHints.isEmpty )
      return
    if ( continuousVariableHints.lengthCompare(continuousVariablesCount) != 0 )
      throw new Exception("Number of continuous variable hints does not correspond to the number of continous variables")
    if ( continuousVariableHints.count(h => h < 1 ) != 0 )
      throw new Exception("Number of suggested normal distributions in a continuous variable must be at least one")
  }

  private def initializeVariableHints(continuousVariableHints: List[Int], continuousVariablesCount: Int): List[Int] = {
    if (continuousVariableHints.isEmpty) {
      var hints = ListBuffer.empty[Int]
      for ( i <- 0 until continuousVariablesCount )
        hints += 1
      hints.toList
    }
    else
      continuousVariableHints
  }
}
