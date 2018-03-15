import java.io.{File, PrintWriter}
import java.util.concurrent.TimeUnit

import com.google.common.io.Files

import scala.util.Random

/**
  * Measures performance (accuracy and time spent learning/testing) on big data set
  */
object Main {

  def main(args: Array[String]): Unit = {
    val temp = Files.createTempDir()
    temp.deleteOnExit()
    val dataSetPath = temp.getPath + "/performance.data"
    generatePerformanceDataSet(dataSetPath, 200, 38, 200, 200, 3)

    val sc = TestUtils.GetSparkContext()
    val perf = Performance.Measure(sc, mle = true, dataSetPath, Option.empty, resourcesPath = false)
    val learningTime = TimeUnit.SECONDS.convert(perf.learningTime, TimeUnit.NANOSECONDS)
    val testingTime = TimeUnit.SECONDS.convert(perf.testingTime, TimeUnit.NANOSECONDS)
    println(s"Average success rate: ${perf.successRate}%")
    println(s"Learning time: $learningTime [s]")
    println(s"Testing time: $testingTime [s]")
  }

  private def generatePerformanceDataSet(dataSetPath: String,
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
