import java.io.{File, PrintWriter}
import java.util.concurrent.TimeUnit

import GaussianUtils.WeightedGaussian
import com.google.common.io.Files
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
  * Measures performance (accuracy and time spent learning/testing) on big data set
  */
object Main {

  def main(args: Array[String]): Unit = {
    val temp = Files.createTempDir()
    temp.deleteOnExit()
    val dataSetPath = temp.getPath + "/performance.data"
    val hints = generatePerformanceDataSet(dataSetPath, 200, 1000, 200, 10, 5, 5, 3, 5)
    //generateLegacyPerformanceDataSet(dataSetPath, 200, 38, 200, 200, 3)

    val sc = TestUtils.GetSparkContext()
    val perf = Performance.Measure(sc, dataSetPath, Option(hints), resourcesPath = false)
    val learningTime = TimeUnit.SECONDS.convert(perf.learningTime, TimeUnit.NANOSECONDS)
    val testingTime = TimeUnit.SECONDS.convert(perf.testingTime, TimeUnit.NANOSECONDS)
    println(s"Average success rate: ${perf.successRate}%")
    println(s"Learning time: $learningTime [s]")
    println(s"Testing time: $testingTime [s]")
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

  def generateDataSet(path: String,
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

  private def generateLegacyPerformanceDataSet(dataSetPath: String,
                                         sequenceLength: Int,
                                         dummyObservedVariableCount: Int,
                                         learningSetLength: Int,
                                         testingSetLength: Int,
                                         worldWidthMultiplier: Int): Unit = {
    val origWorldMatrix = Array("vgyb",
                                "rvgy",
                                "gbrv",
                                "vryb")
    // TODO: it could be better to generate world randomly instead of repeating the same pattern
    val worldMatrix = origWorldMatrix.map(row => (0 until worldWidthMultiplier).map( _ => row)
                                          .foldLeft(""){ (acc,e) => acc + e}) // increse world size
    val height = worldMatrix.length
    val width = worldMatrix(0).length
    val out = new PrintWriter(new File(dataSetPath))
    out.write("discrete discrete continuous")
    (0 until dummyObservedVariableCount).foreach( i => {
      if ( i%2 == 0 )
        out.write(" discrete")
      else
        out.write(" continuous")
    })
    out.write("\n")

    (0 until learningSetLength + testingSetLength).foreach( i => {
      if ( i == learningSetLength )
        out.write("..\n")
      else if ( i != 0 )
        out.write(".\n")

      /* initial position */
      var generated = false
      var row = -1
      var col = -1
      while (!generated) {
        row = Random.nextInt(height)
        col = Random.nextInt(width)
        if (worldMatrix(row)(col) != 'v')
          generated = true
      }
      val color = worldMatrix(row)(col)
      writeState(row, col, color, dummyObservedVariableCount, out, width, height)

      /* steps */
      (0 until sequenceLength).foreach( i => {
        val movements = List(List(-1,0), List(1,0), List(0,1), List(0,-1))
        val idx = Random.nextInt(movements.length)
        val new_row = row + movements(idx)(0).toInt
        val new_col = col + movements(idx)(1).toInt
        if (new_row >= 0 && new_col >= 0 && new_row < height && new_col < width
            && worldMatrix(new_row)(new_col) != 'v') {
          col = new_col
          row = new_row
        }

        var color = worldMatrix(row)(col)
        writeState(row, col, color, dummyObservedVariableCount, out, width, height)
      })
    })

    out.close()
  }

  private def writeState(row: Int, col: Int, color: Char,
                         dummyObservedVariableCount: Int, out: PrintWriter,
                         width: Int, height: Int): Unit = {
    val temperature = TestUtils.GetTemperature(color)
    val coords = (col+1) + ":" + (row+1)
    out.write(coords + " " + TestUtils.GetQuadrant(coords, width, height) + " " + temperature)

    /* write dummy observations */
    (0 until dummyObservedVariableCount).foreach( i => {
      if ( i%2 == 0 )
        out.write(" " + Random.nextInt(100).toString)
      else
        out.write(" " + Random.nextDouble().toString)
    })
    out.write("\n")
  }
}
