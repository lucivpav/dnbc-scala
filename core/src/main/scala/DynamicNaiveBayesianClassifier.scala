import org.apache.spark.SparkContext

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
  * @param initialEdge initial probabilities
  * @param transitions discrete probabilities
  * @param discreteEmissions number of items in the list corresponds to the number of discrete observed variables
  * @param continuousEmissions number of items in the list corresponds to the number of continuous observed variables
  */
class ModelParameters(initialEdge: LearnedDiscreteEdge,
                      transitions: Map[String, LearnedDiscreteEdge],
                      discreteEmissions: List[Map[String, LearnedDiscreteEdge]],
                      continuousEmissions: List[Map[String, LearnedContinuousEdge]]) {
  def InitialEdge = initialEdge
  def Transitions = transitions
  def DiscreteEmissions = discreteEmissions
  def ContinuousEmissions = continuousEmissions
}

/**
  * Learned model used for inference
  * @param parameters
  */
class DynamicNaiveBayesianClassifier(parameters: ModelParameters) {

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
    val discreteVariablesCount = parameters.DiscreteEmissions.length
    val continuousVariablesCount = parameters.ContinuousEmissions.length
    if ( states.exists( s => s.DiscreteVariables.lengthCompare(discreteVariablesCount) != 0 ||
                              s.ContinuousVariables.lengthCompare(continuousVariablesCount) != 0 ) )
      throw new Exception("Number of observed variables is inconsistent")
  }

  private def viterbiInitialize(observedStates: Seq[ObservedState]): Map[String,Double] = {
    val initialEdge = parameters.InitialEdge
    val transitions = parameters.Transitions
    val discreteEmissions = parameters.DiscreteEmissions
    val continuousEmissions = parameters.ContinuousEmissions

    var vcur = Map.empty[String,Double]
    for (hiddenState <- transitions.keys) {
      var emissionsSum = 0.0
      emissionsSum += discreteEmissions.zipWithIndex.map(z => Math.log(z._1(hiddenState)
        .probability(observedStates.head.DiscreteVariables(z._2)))).sum
      emissionsSum += continuousEmissions.zipWithIndex.map(z => Math.log(z._1(hiddenState)
        .probability(observedStates.head.ContinuousVariables(z._2)))).sum
      vcur += (hiddenState -> (emissionsSum + Math.log(initialEdge.probability(hiddenState))))
    }
    vcur
  }

  private def viterbiCompute(probs: Map[String,Double],
                             observedStates: Seq[ObservedState]): List[String] = {
    val initialEdge = parameters.InitialEdge
    val transitions = parameters.Transitions
    val discreteEmissions = parameters.DiscreteEmissions
    val continuousEmissions = parameters.ContinuousEmissions

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
        emissionsSum += discreteEmissions.zipWithIndex.map(z => Math.log(z._1(hiddenState)
                                                            .probability(observedStates(i).DiscreteVariables(z._2)))).sum
        emissionsSum += continuousEmissions.zipWithIndex.map(z => Math.log(z._1(hiddenState)
                                                              .probability(observedStates(i).ContinuousVariables(z._2)))).sum
        vcur += (hiddenState -> (emissionsSum +
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
    learnFinalize(sc, initialEdge, transitions, discreteEmissions, continuousEmissions)
  }

  def baumWelch(sequences: Iterable[Seq[State]]): DynamicNaiveBayesianClassifier = {
    val firstSequence = sequences.take(1)
    val firstDataPoint = firstSequence.toList.head.take(1)
    val discreteVariablesCount = firstDataPoint.head.ObservedState.DiscreteVariables.length
    val continuousVariablesCount = firstDataPoint.head.ObservedState.ContinuousVariables.length
    val originalSequences = (firstSequence ++ sequences).toSeq

    val hiddenStates = List("1:2", "1:3", "2:1", "2:3", "2:4", "3:1", "3:2", "3:3", "3:4", "4:1", "4:2", "4:4") // TODO!

    var possibleTransitions = Map.empty[String,ListBuffer[String]]
    hiddenStates.foreach(hiddenState => possibleTransitions += hiddenState -> ListBuffer.empty[String])
    originalSequences.foreach( seq => {
      (0 until seq.length-1).foreach(i => {
        val from = seq(i)
        val to = seq(i+1)
        if ( !possibleTransitions(from.HiddenState).contains(to.HiddenState) )
          possibleTransitions(from.HiddenState) += to.HiddenState
      })
    })
    var parameters = getInitialModelParameters(hiddenStates, possibleTransitions.map(t => t._1 -> t._2.toList), discreteVariablesCount)

    (0 until 4).foreach(iter => {
      val alphaPointSequences = originalSequences.map(seq => getAlpha(seq.map(s => s.ObservedState), parameters)).toList
      val betas = originalSequences.zipWithIndex.map(z => getBeta(z._1.map(s => s.ObservedState), parameters,
        alphaPointSequences(z._2).map(ap => ap.ScaleFactor))).toList
      val alphas = alphaPointSequences.map(aps => aps.map(ap => ap.Probabilities))

      // sequence | time | from | to
      val gammas = originalSequences.zipWithIndex.map(z => {
        val seq = z._1
        val alpha = alphas(z._2)
        val beta = betas(z._2)
        (0 until seq.length-1).map(t => {
          hiddenStates.map(hiddenStateFrom => {
            val row = hiddenStates.map(hiddenStateTo => {
              var emissionsProd = parameters.DiscreteEmissions.zipWithIndex.map(z => z._1(hiddenStateTo)
                .probability(seq(t+1).ObservedState.DiscreteVariables(z._2))).product
              emissionsProd *= parameters.ContinuousEmissions.zipWithIndex.map(z => z._1(hiddenStateTo)
                .probability(seq(t+1).ObservedState.ContinuousVariables(z._2))).product

              hiddenStateTo -> alpha(t)(hiddenStateFrom) *
                parameters.Transitions(hiddenStateFrom).probability(hiddenStateTo) *
                emissionsProd *
                beta(t+1)(hiddenStateTo)
            }).toMap
            hiddenStateFrom -> row
          }).toMap
        }).toList
      }).toList

      // sequence | time | from
      val deltas = gammas.zipWithIndex.map(sequence => {
        val times = sequence._1.map(time => {
          time.map(from => {
            from._1 -> from._2.values.sum
          })
        })
        val additions = hiddenStates.map(hiddenStateFrom => {
          hiddenStateFrom -> alphas(sequence._2).last(hiddenStateFrom)
        }).toMap
        times ++ List(additions)
      })

      val a = hiddenStates.map(hiddenStateFrom => {
        val row = hiddenStates.map(hiddenStateTo => {
          val a_ij = originalSequences.zipWithIndex.map(sequence => {
            val len = sequence._1.length
            (0 until len-1).map(t => {
              gammas(sequence._2)(t)(hiddenStateFrom)(hiddenStateTo)
            }).sum
          }).sum
          hiddenStateTo -> a_ij
        }).toMap
        hiddenStateFrom -> row
     }).toMap

      /* TODO: discrete emissions */
      val possibleObservations = List("r", "g", "b", "y") //TODO
      val newDiscreteEmissions = (0 until discreteVariablesCount).map(emissionIndex => {
        hiddenStates.map(hiddenState => {
          val p = possibleObservations.map(observation => {

            val top = originalSequences.indices.map(sequenceIndex => {
              val alpha = alphas(sequenceIndex)
              val beta = betas(sequenceIndex)
              val score = alpha.last.values.sum
              val sequence = originalSequences(sequenceIndex)

              val bottomInnerSum = sequence.indices.map(t => {
                val state = sequence(t)
                if (observation == state.ObservedState.DiscreteVariables(emissionIndex))
                  alpha(t)(hiddenState) * beta(t)(hiddenState)
                else
                  0.0
              }).sum

              bottomInnerSum / score
            }).sum

            // TODO: don't duplicate common functionality

            val bottom = originalSequences.indices.map(sequenceIndex => {
              val alpha = alphas(sequenceIndex)
              val beta = betas(sequenceIndex)
              val score = alpha.last.values.sum // P_k
              val sequence = originalSequences(sequenceIndex)

              val bottomInnerSum = sequence.indices.map(t => {
                val state = sequence(t)
                //val observation = state.ObservedState.DiscreteVariables(emissionIndex)
                alpha(t)(hiddenState) * beta(t)(hiddenState)
              }).sum

              bottomInnerSum / score
            }).sum

            val p = top / bottom
            observation -> p
          }).toMap
          hiddenState -> new LearnedDiscreteEdge(p)
        }).toMap
      }).toList

      /* initial edge */
      val pi = hiddenStates.map(hiddenState => {
        val p = originalSequences.indices.map(i => {
          val alpha = alphas(i)
          val beta = betas(i)
          val score = alpha.last.values.sum // P_k

          (alpha.head(hiddenState) * beta.head(hiddenState)) / score
        }).sum
        hiddenState -> p
      }).toMap
      val sum = pi.values.sum
      val normalizedPi = pi.map(z => z._1 -> z._2 / sum) //? ??
      val initialEdge = new LearnedDiscreteEdge(normalizedPi)

      /* transitions */
      val transitions = a.map(t => t._1 -> new LearnedDiscreteEdge(t._2))

      val continuousEmissions = List.empty // TODO
      parameters = new ModelParameters(initialEdge, transitions, newDiscreteEmissions, continuousEmissions)
    })
    new DynamicNaiveBayesianClassifier(parameters)
  }

  private def getInitialModelParameters(possibleHiddenStates: List[String],
                                        possibleTransitions: Map[String,List[String]],
                                        discreteEmissionCount: Int): ModelParameters = {
    /* initial edge */
    val initialEdge = new DiscreteEdge
    possibleHiddenStates.foreach(s => {
      (0 until Random.nextInt(10)+1).foreach(i => initialEdge.learn(s))
    })
    val learnedInitialEdge = initialEdge.learnFinalize()

    /* transitions */
    var transitions = Map.empty[String,LearnedDiscreteEdge]
    possibleTransitions.foreach(t => {
      val from = t._1
      val edge = new DiscreteEdge
      t._2.foreach(to => {
        (0 until Random.nextInt(10)+1).foreach(i => edge.learn(to))
      })
      transitions += from -> edge.learnFinalize()
    })


    /*var transitions = Map.empty[String,DiscreteEdge]
    possibleHiddenStates.foreach(s => {
      val edge = new DiscreteEdge
      possibleHiddenStates.foreach(s => { // I don't think all of these are transitioned into
        edge.learn(s)
        (0 until Random.nextInt(10)).foreach(i => edge.learn(s))
      })
      edge.learnFinalize()
      transitions += s -> edge
    })*/

    /* discrete emissions */
    var discreteEmissions = ListBuffer.empty[Map[String,LearnedDiscreteEdge]]
    (0 until discreteEmissionCount).foreach( i => {
      var p = Map.empty[String, LearnedDiscreteEdge]
      possibleHiddenStates.foreach(s => {
        val edge = new DiscreteEdge
        /* TODO! */
        (0 until Random.nextInt(10)+1).foreach(i => edge.learn("r"))
        (0 until Random.nextInt(10)+1).foreach(i => edge.learn("g"))
        (0 until Random.nextInt(10)+1).foreach(i => edge.learn("b"))
        (0 until Random.nextInt(10)+1).foreach(i => edge.learn("y"))
        p += s -> edge.learnFinalize()
      })
      discreteEmissions += p
    })

    /* continuous emissions - TODO */
    var continuousEmissions = List.empty[Map[String,LearnedContinuousEdge]]

    new ModelParameters(learnedInitialEdge, transitions, discreteEmissions.toList, continuousEmissions)
  }

  private class AlphaPoint(probabilities: Map[String,Double], scaleFactor: Double) {
    def Probabilities = probabilities
    def ScaleFactor = scaleFactor
  }

  // returns forward variable
  private def getAlpha(observations: Seq[ObservedState], parameters: ModelParameters): List[AlphaPoint] = {
    val alpha = ListBuffer.empty[AlphaPoint]

    /* initialization */
    var cur = Map.empty[String,Double]
    parameters.DiscreteEmissions.head.keys.foreach(hiddenState => {
      var emissionsProd = parameters.DiscreteEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observations.head.DiscreteVariables(z._2))).product
      emissionsProd *= parameters.ContinuousEmissions.zipWithIndex.map(z => z._1(hiddenState)
        .probability(observations.head.ContinuousVariables(z._2))).product
      cur += hiddenState -> parameters.InitialEdge.probability(hiddenState) * emissionsProd
    })

    // normalize
    val factor = 1.0/cur.values.sum
    cur = cur.map(z => z._1 -> z._2 * factor)

    alpha += new AlphaPoint(cur, factor)

    /* recursion */
    (1 until observations.length).foreach(i => {
      cur = Map.empty[String,Double]
      alpha.head.Probabilities.keys.foreach(hiddenState1 => {
        var emissionsProd = parameters.DiscreteEmissions.zipWithIndex.map(z => z._1(hiddenState1)
          .probability(observations(i).DiscreteVariables(z._2))).product
        emissionsProd *= parameters.ContinuousEmissions.zipWithIndex.map(z => z._1(hiddenState1)
          .probability(observations(i).ContinuousVariables(z._2))).product

        val p = alpha.head.Probabilities.keys.map(hiddenState2 => {
          alpha(i-1).Probabilities(hiddenState2) * parameters.Transitions(hiddenState2).probability(hiddenState1)
        }).sum * emissionsProd
        cur += hiddenState1 -> p
      })

      // normalize
      val factor = 1.0/cur.values.sum
      cur = cur.map(z => z._1 -> z._2 * factor)

      alpha += new AlphaPoint(cur, factor)
    })

    alpha.toList
  }

  private def getBeta(observations: Seq[ObservedState], parameters: ModelParameters,
                      scaleFactors: List[Double]): List[Map[String,Double]] = {
    if ( observations.length != scaleFactors.length )
      throw new Exception("Incorrect number of scale factors or observations provided")
    val beta = ListBuffer.empty[Map[String,Double]]

    /* initialization */
    var cur = Map.empty[String,Double]
    parameters.DiscreteEmissions.head.keys.foreach(hiddenState => {
      cur += hiddenState -> 1
    })

    // normalize
    cur = cur.map(z => z._1 -> z._2 * scaleFactors.last)

    beta += cur

    /* recursion */
    (1 until observations.length).foreach(i => {
      cur = Map.empty[String,Double]
      beta.head.keys.foreach(hiddenState1 => {
        val p = beta.head.keys.map(hiddenState2 => {
          val observationIndex = observations.length-i
          var emissionsProd = parameters.DiscreteEmissions.zipWithIndex.map(z => z._1(hiddenState2)
            .probability(observations(observationIndex).DiscreteVariables(z._2))).product
          emissionsProd *= parameters.ContinuousEmissions.zipWithIndex.map(z => z._1(hiddenState2)
            .probability(observations(observationIndex).ContinuousVariables(z._2))).product
          parameters.Transitions(hiddenState1).probability(hiddenState2) * emissionsProd * beta(i-1)(hiddenState2)
        }).sum
        cur += hiddenState1 -> p
      })

      // normalize
      cur = cur.map(z => z._1 -> z._2 * scaleFactors(scaleFactors.length-1-i))

      beta += cur
    })
    //beta.foreach(b => print(b.head))
    //println()
    beta.reverse.toList
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

  private def learnFinalize(sc: SparkContext,
                            initialEdge: DiscreteEdge, transitions: Map[String, DiscreteEdge],
                            discreteEmissions: Array[Map[String, DiscreteEdge]],
                            continuousEmissions: Array[Map[String, ContinuousEdge]]): DynamicNaiveBayesianClassifier = {
    /* finalize initial edge */
    val learnedInitialEdge = initialEdge.learnFinalize()

    /* finalize transitions */
    val parallelTransitions = sc.parallelize(transitions.toSeq)
    val learnedTransitionsArray = parallelTransitions.map(t => t._1 -> t._2.learnFinalize()).collect()
    var learnedTransitionsMap = Map.empty[String,LearnedDiscreteEdge]
    learnedTransitionsArray.foreach(p => learnedTransitionsMap += p)

    /* finalize discrete emissions */
    val parallelDiscreteEmissions = sc.parallelize(discreteEmissions)
    val learnedDiscreteEmissions = parallelDiscreteEmissions.map(m => m.map(p => p._1 -> p._2.learnFinalize()))
                                                                  .collect().toList

    /* finalize continuous emissions */
    // The reason I am not using SparkContext.parallelize is that the learnFinalize() function itself creates RDD.
    // Nested RDDs are not allowed in Spark
    val learnedContinuousEmissions = continuousEmissions.par.map(m => m.par.map(z => z._1 -> z._2.learnFinalize()).seq)
                                                                  .toList

    new DynamicNaiveBayesianClassifier(new ModelParameters(learnedInitialEdge, learnedTransitionsMap,
                                        learnedDiscreteEmissions, learnedContinuousEmissions))
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
