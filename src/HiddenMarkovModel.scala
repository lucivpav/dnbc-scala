// Usage: call learnSequence() once per each data in training set
//        then finalize learning stage by calling learnFinalize()
class HiddenMarkovModel(sequenceLength: Int) {
  private var transitionCounts: Option[Array[Array[Int]]] = None
  private var emissionCounts: Option[Array[Array[Int]]] = None

  private var hiddenStateMap: Map[String, Int] = Map.empty
  private var observedStateMap: Map[String, Int] = Map.empty
  private var hiddenStateLastIndex: Int = 0
  private var observedStateLastIndex: Int = 0

  private var transitions: Array[Array[Double]] = Array.empty
  private var emissions: Array[Array[Double]] = Array.empty

  private var initialCounts: Array[Int] = Array.empty
  private var initialProbability: Array[Double] = Array.empty

  def learnSequence(hiddenStates: List[String], observedStates: List[String]): Unit = {
    if ( hiddenStates.length != sequenceLength || observedStates.length != sequenceLength )
      throw new Exception("Sequence to be learned is longer than expected sequence length")

    addNewStatesToStateMaps(hiddenStates, observedStates)

    resizeTransitionCountsIfNecessary()
    resizeEmissionCountsIfNecessary()
    resizeInitialCountsIfNecessary()

    addTransitionCounts(hiddenStates)
    addEmissionCounts(hiddenStates, observedStates)
    addInitialCounts(hiddenStates)
  }

  def learnFinalize(): Unit = {
    calculateTransitions()
    calculateEmissions()
    calculateInitialProbability()
  }

  private def calculateTransitions(): Unit = {
    transitions = Array.ofDim[Double](hiddenStateLastIndex, hiddenStateLastIndex)
    for (i <- transitions.indices)
      for (j <- transitions(i).indices) {
        var sum = 0
        transitionCounts.get(i).foreach(sum += _)
        transitions(i)(j) = transitionCounts.get(i)(j).toDouble / sum
      }
  }

  private def calculateEmissions(): Unit = {
    emissions = Array.ofDim[Double](hiddenStateLastIndex, observedStateLastIndex)
    for (i <- emissions.indices)
      for (j <- emissions(i).indices) {
        var sum = 0
        emissionCounts.get(i).foreach(sum += _)
        emissions(i)(j) = emissionCounts.get(i)(j).toDouble / sum
      }
  }

  private def calculateInitialProbability(): Unit = {
    val sum = initialCounts.sum
    initialProbability = Array.ofDim(hiddenStateLastIndex)
    for ( i <- initialProbability.indices )
      initialProbability(i) = initialCounts(i).toDouble / sum
  }

  private def addTransitionCounts(hiddenStates: List[String]): Unit = {
    for (i <- 0 to sequenceLength - 2) {
      val from = hiddenStateMap(hiddenStates(i))
      val to = hiddenStateMap(hiddenStates(i + 1))
      transitionCounts.get(from)(to) += 1
    }
  }

  private def addEmissionCounts(hiddenStates: List[String], observedStates: List[String]): Unit = {
    for (i <- 0 until sequenceLength) {
      val from = hiddenStateMap(hiddenStates(i))
      val to = observedStateMap(observedStates(i))
      emissionCounts.get(from)(to) += 1
    }
  }

  private def addInitialCounts(hiddenStates: List[String]): Unit = {
    initialCounts(hiddenStateMap(hiddenStates.head)) += 1
  }

  private def resizeEmissionCountsIfNecessary(): Unit = {
    emissionCounts match {
      case None => emissionCounts = Option(Array.ofDim[Int](hiddenStateLastIndex, observedStateLastIndex))
      case _ if emissionCounts.get.length != hiddenStateLastIndex || emissionCounts.get(1).length != observedStateLastIndex =>
        val prev = emissionCounts.get.clone()
        emissionCounts = Option(Array.ofDim[Int](hiddenStateLastIndex, observedStateLastIndex))
        for (i <- prev.indices)
          for (j <- prev(i).indices)
            emissionCounts.get(i)(j) = prev(i)(j)
      case _ =>
    }
  }

  private def resizeTransitionCountsIfNecessary(): Unit = {
    transitionCounts match {
      case None => transitionCounts = Option(Array.ofDim[Int](hiddenStateLastIndex, hiddenStateLastIndex))
      case _ if transitionCounts.get.length != hiddenStateLastIndex =>
        val prev = transitionCounts.get.clone()
        transitionCounts = Option(Array.ofDim[Int](hiddenStateLastIndex, hiddenStateLastIndex))
        for (i <- prev.indices)
          for (j <- prev(i).indices)
            transitionCounts.get(i)(j) = prev(i)(j)
      case _ =>
    }
  }

  private def resizeInitialCountsIfNecessary(): Unit = {
    val prev = initialCounts.clone()
    initialCounts = Array.ofDim[Int](hiddenStateLastIndex)
    for (i <- prev.indices)
        initialCounts(i) = prev(i)
  }

  private def addNewStatesToStateMaps(hiddenStates: List[String], observedStates: List[String]): Unit = {
    for (hiddenState <- hiddenStates) {
      if (!hiddenStateMap.contains(hiddenState)) {
        hiddenStateMap += hiddenState -> hiddenStateLastIndex
        hiddenStateLastIndex += 1
      }
    }

    for (observedState <- observedStates) {
      if (!observedStateMap.contains(observedState)) {
        observedStateMap += observedState -> observedStateLastIndex
        observedStateLastIndex += 1
      }
    }
  }

}
