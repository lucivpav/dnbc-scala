import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

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
}
