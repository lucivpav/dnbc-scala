import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.io.Source
import scala.util.Random

object Main {

  def main(args: Array[String]): Unit = {
    val continuousDatasetName = "dataset/robot_no_momentum_continuous.data"
    GenerateContinuousDataSet(continuousDatasetName)
    GenerateDataSetWithTwoVariables(continuousDatasetName)
    GenerateDataSetWithGaussianMixtureVariable("dataset/gaussian_mixture.data")
  }

  // Generates continuous data set based on robot_no_momentum.data
  private def GenerateContinuousDataSet(continuousDataSetName: String): Unit = {
    var in = Source.fromFile("dataset/robot_no_momentum.data")
    var out = new PrintWriter(new File(continuousDataSetName))

    for (line <- in.getLines()) {
      if (line == "." || line == "..") {
        out.write(line + "\n")
      }
      else {
        val hiddenState = line.substring(0, 3)
        val observedDiscreteState = line.substring(4)
        val observedContinuousState = observedDiscreteState match {
          case "r" => GaussianUtils.nextGaussian(20,25)
          case "g" => GaussianUtils.nextGaussian(35, 49)
          case "b" => GaussianUtils.nextGaussian(15, 9)
          case "y" => GaussianUtils.nextGaussian(5, 1)
        }
        out.write(hiddenState + " " + observedContinuousState + "\n")
      }
    }
    out.close()
  }

  // Generates data set based on previously generated continuous data set
  // The resulting data set contains two observed variables, one discrete and one continuous
  private def GenerateDataSetWithTwoVariables(continuousDataSetName: String): Unit = {
    var in = Source.fromFile(continuousDataSetName)
    var out = new PrintWriter(new File("dataset/robot_no_momentum_bivariate.data"))
    for (line <- in.getLines()) {
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
    var out = new PrintWriter(new File(dataSetName))
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
