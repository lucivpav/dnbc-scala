import scala.collection.mutable.ListBuffer

// Usage: call learnSequence() once per each data in training set
//        then finalize learning stage by calling learnFinalize()
class DynamicNaiveBayesianClassifier(sequenceLength: Int,
                                     discreteVariablesCount: Int, continuousVariablesCount: Int) {
  private var initialEdge: DiscreteEdge = new DiscreteEdge
  private var transitions: Map[String, DiscreteEdge] = Map.empty
  private var discreteEmissions: Map[String, DiscreteEdge] = Map.empty
  private var continuousEmissions: Map[String, ContinuousEdge] = Map.empty

  def learnSequence(hiddenStates: List[String],
                    observedDiscreteVariables: List[List[String]],
                    observedContinuousVariables: List[List[Double]]): Unit = {
    if ( observedDiscreteVariables.length != discreteVariablesCount )
      throw new Exception("Number of observed discrete variables does not correspond to count set")
    if ( observedContinuousVariables.length != continuousVariablesCount )
      throw new Exception("Number of observed continuous variables does not correspond to count set")

    if ( discreteVariablesCount+continuousVariablesCount != 1 )
      throw new Exception("Support for multiple observed variables not yet implemented")

    val discrete = discreteVariablesCount == 1
    var observedDiscreteStates = List.empty[String]
    var observedContinuousStates = List.empty[Double]
    if ( discrete )
      observedDiscreteStates ++= observedDiscreteVariables.head
    else
      observedContinuousStates ++= observedContinuousVariables.head

    if ( hiddenStates.length != sequenceLength ||
          (discrete && observedDiscreteStates.length != sequenceLength ) ||
          (!discrete && observedContinuousStates.length != sequenceLength) )
      throw new Exception("Sequence to be learned is longer than expected sequence length")

    /* discrete emissions */
    if (discrete) {
      for (i <- hiddenStates.indices) {
        val hiddenState = hiddenStates(i)
        val observedState = observedDiscreteStates(i)
        if (!discreteEmissions.contains(hiddenState))
          discreteEmissions += (hiddenState -> new DiscreteEdge)
        discreteEmissions(hiddenState).learn(observedState)
      }
    }
    else { /* continuous emissions */
      for (i <- hiddenStates.indices) {
        val hiddenState = hiddenStates(i)
        val observedState = observedContinuousStates(i)
        if (!continuousEmissions.contains(hiddenState))
          continuousEmissions += (hiddenState -> new ContinuousEdge())
        continuousEmissions(hiddenState).learn(observedState)
      }
    }

    /* transitions */
    for (i <- 0 to sequenceLength - 2) {
      val from = hiddenStates(i)
      val to = hiddenStates(i+1)
      if (!transitions.contains(from))
        transitions += (from -> new DiscreteEdge)
      transitions(from).learn(to)
    }
    initialEdge.learn(hiddenStates.head)
  }

  def learnFinalize(): Unit = {
    initialEdge.learnFinalize()
    transitions.foreach(t => t._2.learnFinalize())
    discreteEmissions.foreach(e => e._2.learnFinalize())
    continuousEmissions.foreach(e => e._2.learnFinalize())
  }

  def infereMostLikelyHiddenStates(observedDiscreteVariables: List[List[String]],
                                    observedContinuousVariables: List[List[Double]]): List[String] = {
    val discrete = discreteVariablesCount == 1
    var observedDiscreteStates = List.empty[String]
    var observedContinuousStates = List.empty[Double]
    if (discrete)
      observedDiscreteStates ++= observedDiscreteVariables.head
    else
      observedContinuousStates ++= observedContinuousVariables.head

    // Viterbi

    /* initialization */
    var path = ListBuffer.empty[String]
    var vprev = Map.empty[String,Double]
    var vcur = Map.empty[String,Double]
    for ( hiddenState <- transitions.keys )
    {
      var emissionsSum = 0.0
      if (discrete)
        emissionsSum += discreteEmissions(hiddenState).probability(observedDiscreteStates.head)
      else
        emissionsSum += continuousEmissions(hiddenState).probability(observedContinuousStates.head)
      vcur += (hiddenState -> (Math.log(emissionsSum) + Math.log(initialEdge.probability(hiddenState))))
    }
    vprev = vcur

    /* algorithm */
    for( i <- 1 until sequenceLength )
    {
      // append path
      path += vprev.maxBy(_._2)._1
      var vcur = Map.empty[String,Double]
      for ( hiddenState <- transitions.keys )
      {
        var emissionsSum = 0.0
        if (discrete)
          emissionsSum += discreteEmissions(hiddenState).probability(observedDiscreteStates(i))
        else
          emissionsSum += continuousEmissions(hiddenState).probability(observedContinuousStates(i))
        vcur += (hiddenState -> (Math.log(emissionsSum) +
                                  vprev.maxBy{case (hs,p) => p + Math.log(transitions(hs).probability(hiddenState))}._2))
      }
      vprev = vcur
    }
    path += vprev.maxBy(_._2)._1
    path.toList
  }
}
