import java.io.{File, PrintWriter}
import java.util.concurrent.TimeUnit

import GaussianUtils.WeightedGaussian
import com.google.common.io.Files
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

/**
  * Measures performance (accuracy and time spent learning/testing) on big data set
  */
object Main {

  def main(args: Array[String]): Unit = {
    val temp = Files.createTempDir()
    temp.deleteOnExit()
    val dataSetPath = temp.getPath + "/performance.data"

    if ( args.length == 0 )
    {
      val sc = TestUtils.GetSparkContext(2)
      generateAndMeasure(sc, dataSetPath, 200, 1000, 200, 10, 5, 5, 3, 5)
      return
    }

    if ( args.length != 9 && args.length != 8 )
    {
        println("unsupported configuration")
        return
    }
    println("assumed structure:")
    val clusterMode = args.length == 8
    if ( !clusterMode )
      print("workers ")
    println("sequenceLength learningSetLength testingSetLength hiddenStateCount" +
            " discreteEmissionCount continuousEmissionCount maxGaussiansPerMixture transitionsPerNode")
    println("config:")
    println(args.mkString(" "))
    var s = args.map(a => a.toInt)
    val sc = if (clusterMode) SparkContext.getOrCreate() else TestUtils.GetSparkContext(s(0))
    if ( clusterMode )
      s = (List(-1) ++ s).toArray
    try {
      generateAndMeasure(sc, dataSetPath, s(1), s(2), s(3), s(4), s(5), s(6), s(7), s(8))
    } catch {
      case e: Exception => println(e)
    }
    sc.stop()
  }

  def generateAndMeasure(sc: SparkContext, path: String, sequenceLength: Int, learningSetLength: Int, testingSetLength: Int,
              hiddenStateCount: Int, discreteEmissionCount: Int, continuousEmissionCount: Int,
              maxGaussiansPerMixture: Int, transitionsPerNode: Int): Unit = {

    val hints = TestUtils.generatePerformanceDataSet(path, sequenceLength, learningSetLength,testingSetLength, hiddenStateCount,
                            discreteEmissionCount, continuousEmissionCount, maxGaussiansPerMixture, transitionsPerNode)
    //TestUtils.generateLegacyPerformanceDataSet(dataSetPath, 200, 38, 200, 200, 3)
    val perf = Performance.Measure(sc, path, Option(hints), resourcesPath = false)
    val learningTime = TimeUnit.SECONDS.convert(perf.learningTime, TimeUnit.NANOSECONDS)
    val testingTime = TimeUnit.SECONDS.convert(perf.testingTime, TimeUnit.NANOSECONDS)
    println(s"Average success rate: ${perf.successRate}%")
    println(s"Learning time: $learningTime [s]")
    println(s"Testing time: $testingTime [s]")
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
