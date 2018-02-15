import java.io.{File, PrintWriter}

import scala.io.Source
import scala.util.Random

object Main {
  def main(args: Array[String]): Unit = {
    val continuousDatasetName = "dataset/robot_no_momemtum_continuous.data"
    GenerateContinuousDataSet(continuousDatasetName)
    GenerateDataSetWithTwoVariables(continuousDatasetName)
  }

  // Generates continuous data set based on robot_no_momentum.data
  private def GenerateContinuousDataSet(continuousDataSetName: String): Unit = {
    var in = Source.fromFile("dataset/robot_no_momemtum.data")
    var out = new PrintWriter(new File(continuousDataSetName))

    val RedGaussian = new RandomGaussian(20, 5)
    val GreenGaussian = new RandomGaussian(35, 7)
    val BlueGaussian = new RandomGaussian(15, 3)
    val YellowGaussian = new RandomGaussian(5, 1)

    for (line <- in.getLines()) {
      if (line == "." || line == "..") {
        out.write(line + "\n")
      }
      else {
        val hiddenState = line.substring(0, 3)
        val observedDiscreteState = line.substring(4)
        val observedContinuousState = observedDiscreteState match {
          case "r" => RedGaussian.nextRandom()
          case "g" => GreenGaussian.nextRandom()
          case "b" => BlueGaussian.nextRandom()
          case "y" => YellowGaussian.nextRandom()
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
    var out = new PrintWriter(new File("dataset/robot_no_momemtum_bivariate.data"))
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
}

class RandomGaussian(mean: Double, variance: Double) {
  def nextRandom(): Double = {
    Random.nextGaussian()*variance+mean
  }
}
