import java.io.{File, PrintWriter}

import GaussianUtils.WeightedGaussian
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.Random

object TestUtils {

  /**
    * @return newly created SparkContext
    */
  def GetSparkContext(workers: Int): SparkContext = { // TODO: adjust for use in cluster
    val sc = SparkSession.builder.appName("Simple Application").config("spark.master", s"local[$workers]")
                                .getOrCreate().sparkContext
    sc.setLogLevel("ERROR")
    sc
  }

  /**
    * Returned quadrant can be assigned incorrectly with respect to predefined error rate
    * @param hiddenState coordinates, e.g. "4:13"
    * @param width width of world
    * @param height height of world
    * @return quadrant number, the world rectangle is split into four parts:
    *         +-+-+
    *         |2|1|
    *         +-+-+
    *         |3|4|
    *         +-+-+
    */
  def GetQuadrant(hiddenState: String, width: Int = 4, height: Int = 4): String = {
    val pError = 0.15
    val splitted = hiddenState.split(":")
    val x = splitted(0).toInt
    val y = splitted(1).toInt
    var quadrate = 0

    if (x <= width / 2) {
      if (y <= height / 2)
        quadrate = 3
      else
        quadrate = 2
    }
    else {
      if (y <= height / 2)
        quadrate = 4
      else
        quadrate = 1
    }

    if (Random.nextDouble() < pError)
      quadrate = Random.nextInt(4) + 1
    quadrate.toString
  }

  /**
    * Each color has its own surface temperature distribution
    * @param color red, green, blue or yellow
    * @return random temperature associated with color
    */
  def GetTemperature(color: Char): Double = {
    color match {
      case 'r' => GaussianUtils.nextGaussian(20,25)
      case 'g' => GaussianUtils.nextGaussian(35, 49)
      case 'b' => GaussianUtils.nextGaussian(15, 9)
      case 'y' => GaussianUtils.nextGaussian(5, 1)
    }
  }

  def generatePerformanceDataSet(path: String, sequenceLength: Int, learningSetLength: Int, testingSetLength: Int,
              hiddenStateCount: Int, discreteEmissionCount: Int, continuousEmissionCount: Int,
              maxGaussiansPerMixture: Int, transitionsPerNode: Int): List[Int] = {

    val hiddenStates = (0 until hiddenStateCount).map(i => "h" + i.toString).toList
    val initialEdge = getInitialEdge(hiddenStates)
    val transitions = getTransitions(transitionsPerNode, hiddenStates)
    val discreteEmissions = getDiscreteEmissions(discreteEmissionCount, hiddenStates)
    val continuousVariables = getContinuousEmissions(continuousEmissionCount, maxGaussiansPerMixture, hiddenStates)

    generateDataSet(path, sequenceLength, learningSetLength, testingSetLength,
      initialEdge, transitions, discreteEmissions, continuousVariables.map(e => e.Emissions))
    continuousVariables.map(e => e.Gaussians)
  }

  private def getInitialEdge(hiddenStates: List[String]): RandomDiscreteEdge = {
    val initialProbabilities = hiddenStates.map(hiddenState => hiddenState -> GaussianUtils.nextGaussian(100, 16)).toMap
    val sum = initialProbabilities.values.sum
    val normalizedInitialProbabilities = initialProbabilities.map(p => p._1 -> p._2 / sum)
    new RandomDiscreteEdge(normalizedInitialProbabilities)
  }

  private def getTransitions(edgesPerNode: Int, hiddenStates: List[String]): Map[String,RandomDiscreteEdge] = {
    hiddenStates.map(hiddenState => {
      var possibleDestinations = hiddenStates.toList
      val destinations = ListBuffer.empty[String]
      (0 until edgesPerNode).foreach(i => {
        val dIdx = Random.nextInt(possibleDestinations.length)
        destinations += possibleDestinations(dIdx)
        possibleDestinations = possibleDestinations.patch(dIdx, Nil, 1)
      })
      val probabilities = destinations.map(d => d -> Random.nextDouble()).toMap
      val sum = probabilities.values.sum
      val normalizedProbabilities = probabilities.map(p => p._1 -> p._2 / sum)
      hiddenState -> new RandomDiscreteEdge(normalizedProbabilities)
    }).toMap
  }

  private def getDiscreteEmissions(emissionCount: Int,
                                   hiddenStates: List[String]): List[Map[String,RandomDiscreteEdge]] = {
    (0 until emissionCount).map(_ => {
      hiddenStates.map(hiddenState => {
        val probabilities = (0 until 10).map(i => {
          i.toString -> Random.nextDouble()
        }).toMap
        val sum = probabilities.values.sum
        val normalizedProbabilities = probabilities.map(p => p._1 -> p._2 / sum)
        hiddenState -> new RandomDiscreteEdge(normalizedProbabilities)
      }).toMap
    }).toList
  }

  private class ContinuousVariable(emissions: Map[String,RandomContinuousEdge], gaussians: Int) {
    def Emissions = emissions
    def Gaussians = gaussians
  }

  private def getContinuousEmissions(emissionCount: Int, maxGaussiansPerMixture: Int,
                                     hiddenStates: List[String]): List[ContinuousVariable] = {
    val gaussianCounts = (0 until emissionCount).map(_ => 1+Random.nextInt(maxGaussiansPerMixture+1)).toList
    (0 until emissionCount).map(i => {
      val emissions = hiddenStates.map(hiddenState => {
        val gaussians = (0 until gaussianCounts(i)).map(_ => {
          val minMean = -10
          val maxMean = 10
          val mean = Random.nextInt(maxMean - minMean + 1) + minMean

          val minSigma = 1
          val maxSigma = 4
          val sigma = Random.nextInt(maxSigma - minSigma + 1) + minSigma

          new MultivariateGaussian(Vectors.dense(mean), Matrices.dense(1,1, Array(sigma)))
        }).toList
        val weight = 1.0/gaussians.length
        hiddenState -> new RandomContinuousEdge(gaussians.map(g => new WeightedGaussian(weight,g)))
      }).toMap
      new ContinuousVariable(emissions, gaussianCounts(i))
    }).toList
  }

  private def generateDataSet(path: String,
                      sequenceLength: Int,
                      learningSetLength: Int,
                      testingSetLength: Int,
                      initialEdge: RandomDiscreteEdge,
                      transitions: Map[String, RandomDiscreteEdge],
                      discreteEmissions: List[Map[String, RandomDiscreteEdge]],
                      continuousEmissions: List[Map[String, RandomContinuousEdge]]): Unit = {
    val out = new PrintWriter(new File(path))

    /* header */
    out.write("discrete")
    out.write(discreteEmissions.map(e => " discrete").fold("")((a,b) => a + b))
    out.write(continuousEmissions.map(e => " continuous").fold("")((a,b) => a + b))
    out.write("\n")

    (0 until learningSetLength+testingSetLength).foreach(i => {
      if ( i == learningSetLength )
        out.write("..\n")
      else if ( i != 0 )
        out.write(".\n")
      var prevHiddenState = ""
      (0 until sequenceLength).foreach(j => {
        val hiddenState = if (j==0) initialEdge.next() else transitions(prevHiddenState).next()
        out.write(hiddenState)
        out.write(discreteEmissions.map(e => " " + e(hiddenState).next()).fold("")((a,b) => a+b))
        out.write(continuousEmissions.map(e => " " + e(hiddenState).next()).fold("")((a,b) => a+b))
        out.write("\n")
        prevHiddenState = hiddenState
      })
    })

    out.close()
  }

}
