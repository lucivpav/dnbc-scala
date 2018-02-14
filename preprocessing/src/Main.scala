import java.io.{File, PrintWriter}

import scala.io.Source
import scala.util.Random

// Generates continuous data set based on robot_no_momentum.data
object Main {
  def main(args: Array[String]): Unit = {

    var in = Source.fromFile("dataset/robot_no_momemtum.data")
    var out = new PrintWriter(new File("dataset/robot_no_momemtum_continuous.data"))

    val RedGaussian = new RandomGaussian(20, 5)
    val GreenGaussian = new RandomGaussian(35, 7)
    val BlueGaussian = new RandomGaussian(15, 3)
    val YellowGaussian = new RandomGaussian(5, 1)

    for ( line <- in.getLines() ) {
      if ( line == "." || line == ".." ) {
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
}

class RandomGaussian(mean: Double, variance: Double) {
  def nextRandom(): Double = {
    Random.nextGaussian()*variance+mean
  }
}
