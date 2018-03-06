import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.io.Source
import scala.util.Random

/**
  * Generates data sets used by core-test unit tests
  */
object Main {
  private val dataSetDirectory = "core-test/src/test/resources"

  def main(args: Array[String]): Unit = {
    val continuousDatasetName = "robot_no_momentum_continuous.data"
    GenerateContinuousDataSet(continuousDatasetName)
    GenerateDataSetWithTwoVariables(continuousDatasetName)
    GenerateDataSetWithGaussianMixtureVariable("gaussian_mixture.data")
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
        val observedContinuousState = TestUtils.GetTemperature(observedDiscreteState(0))
        out.write(hiddenState + " " + observedContinuousState + "\n")
      }
    }
    out.close()
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
        val observedDiscreteState = TestUtils.GetQuadrant(hiddenState)
        out.write(hiddenState + " " + observedContinuousState + " " + observedDiscreteState + "\n")
      }
    }
    out.close()
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
