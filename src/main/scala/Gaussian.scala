import scala.util.Random

class Gaussian(mean: Double, variance: Double) {

  def nextRandom(): Double = {
    Random.nextGaussian()*scala.math.sqrt(variance)+mean
  }

  def density(x: Double): Double = {
    (1/scala.math.sqrt(2*scala.math.Pi*variance))*scala.math.pow(scala.math.E,-scala.math.pow(x - mean, 2)/(2*variance))
  }
}
