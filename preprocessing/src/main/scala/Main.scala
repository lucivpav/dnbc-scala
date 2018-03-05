import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.io.Source
import scala.util.Random

object Main {
  private val dataSetDirectory = "core/src/test/resources"

  def main(args: Array[String]): Unit = {
    //val continuousDatasetName = "robot_no_momentum_continuous.data"
    //GenerateContinuousDataSet(continuousDatasetName)
    //GenerateDataSetWithTwoVariables(continuousDatasetName)
    //GenerateDataSetWithGaussianMixtureVariable("gaussian_mixture.data")
    GeneratePerformanceDataSet("performance.data", 200, 16, 200, 200)
  }

  private def GeneratePerformanceDataSet(dataSetName: String,
                                         sequenceLength: Int,
                                         dummyObservedVariableCount: Int,
                                         learningSetLength: Int,
                                         testingSetLength: Int): Unit = {
    new File("dataset").mkdir()
    val origWorldMatrix = Array("vgyb",
                                "rvgy",
                                "gbrv",
                                "vryb")
    val worldMatrix = origWorldMatrix.map(row => row + row) // increse world size
    val height = worldMatrix.length
    val width = worldMatrix(0).length
    val out = new PrintWriter(new File("dataset/" + dataSetName))
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
      writeState(row, col, color, dummyObservedVariableCount, out)

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
        writeState(row, col, color, dummyObservedVariableCount, out)
      })
    })

    out.close()
  }

  private def writeState(row: Int, col: Int, color: Char, dummyObservedVariableCount: Int, out: PrintWriter): Unit = {
    val temperature = getTemperatureAssociatedWithColor(color)
    val coords = (col+1) + ":" + (row+1)
    out.write(coords + " " + getQuadrate(coords) + " " + temperature)

    /* write dummy observations */
    (0 until dummyObservedVariableCount).foreach( i => {
      if ( i%2 == 0 )
        out.write(" " + Random.nextInt(100).toString)
      else
        out.write(" " + Random.nextDouble().toString)
    })
    out.write("\n")
  }

  // Generates continuous data set based on robot_no_momentum.data
  private def GenerateContinuousDataSet(continuousDataSetName: String): Unit = {
    val in = Source.fromFile(dataSetDirectory + "/robot_no_momentum.data")
    var out = new PrintWriter(new File(dataSetDirectory + "/" + continuousDataSetName))
    out.write("discrete continuous\n")
    for (line <- in.getLines().drop(1)) {
      if (line == "." || line == "..") {
        out.write(line + "\n")
      }
      else {
        val hiddenState = line.substring(0, 3)
        val observedDiscreteState = line.substring(4)
        val observedContinuousState = getTemperatureAssociatedWithColor(observedDiscreteState(0))
        out.write(hiddenState + " " + observedContinuousState + "\n")
      }
    }
    out.close()
  }

  private def getTemperatureAssociatedWithColor(color: Char): Double = {
    color match {
      case 'r' => GaussianUtils.nextGaussian(20,25)
      case 'g' => GaussianUtils.nextGaussian(35, 49)
      case 'b' => GaussianUtils.nextGaussian(15, 9)
      case 'y' => GaussianUtils.nextGaussian(5, 1)
    }
  }

  // Generates data set based on previously generated continuous data set
  // The resulting data set contains two observed variables, one discrete and one continuous
  private def GenerateDataSetWithTwoVariables(continuousDataSetName: String): Unit = {
    val in = Source.fromFile(dataSetDirectory + "/" + continuousDataSetName)
    var out = new PrintWriter(new File(dataSetDirectory + "/robot_no_momentum_bivariate.data"))
    out.write("discrete continuous discrete\n")
    for (line <- in.getLines().drop(1)) {
      if (line == "." || line == "..") {
        out.write(line + "\n")
      }
      else {
        val hiddenState = line.substring(0, 3)
        val observedContinuousState = line.substring(4)
        val observedDiscreteState = getQuadrate(hiddenState)
        out.write(hiddenState + " " + observedContinuousState + " " + observedDiscreteState + "\n")
      }
    }
    out.close()
  }

  // +-+-+
  // |2|1|
  // +-+-+
  // |3|4|
  // +-+-+
  private def getQuadrate(hiddenState: String): String = {
    val pError = 0.15
    val width = 4
    val height = 4
    val x = hiddenState.substring(0,1).toInt
    val y = hiddenState.substring(2,3).toInt
    var quadrate = 0

    if ( x <= width/2 ) {
      if ( y <= height/2 )
        quadrate = 3
      else
        quadrate = 2
    }
    else {
      if ( y <= height/2 )
        quadrate = 4
      else
        quadrate = 1
    }

    if ( Random.nextDouble() < pError )
      quadrate = Random.nextInt(4)+1
    quadrate.toString
  }

  private def GenerateDataSetWithGaussianMixtureVariable(dataSetName: String): Unit = {
    var out = new PrintWriter(new File(dataSetDirectory + "/" + dataSetName))
    out.write("discrete continuous\n")
    val rnd = new Random()
    val gaussians = List(new MultivariateGaussian(Vectors.dense(-20), Matrices.dense(1,1, Array(9))),
                          new MultivariateGaussian(Vectors.dense(20), Matrices.dense(1,1, Array(9))))
    for ( i <- 0 until 2001 ) {
      val hiddenState = rnd.nextBoolean()
      var value = 0.0

      if ( hiddenState ) value = GaussianUtils.nextGaussian(0,16)
      else value = GaussianUtils.nextGaussianMixture(gaussians)

      if ( i == 1000 ) out.write("..\n") // separate training and testing data
      else out.write(hiddenState + " " + value + "\n")
    }
    out.close()
  }
}
