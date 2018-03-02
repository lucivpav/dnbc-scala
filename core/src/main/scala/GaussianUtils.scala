import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.util.Random

/**
  * Utility functions that deal with normal distribution and related topics
  */
object GaussianUtils {

  /**
    * Returns a randomly generated number from given normal distribution
    * @param mean mean of normal distribution
    * @param variance variance of normal distribution
    * @return random number that corresponds to given normal distribution
    */
  def nextGaussian(mean: Double, variance: Double): Double = {
    Random.nextGaussian()*scala.math.sqrt(variance)+mean
  }

  /**
    * Returns a randomly generated number from gaussian mixture distribution
    * @param gaussians normal distributions representing a gaussian mixture, only one variable per gaussian is supported
    * @return random number that corresponds to given gaussian mixture distribution
    */
  def nextGaussianMixture(gaussians: List[MultivariateGaussian]): Double = {
    var generated = false
    var random = 0.0
    while (!generated) {
      val i = Random.nextInt(gaussians.length)
      random = nextGaussian(gaussians(i).mu(0), gaussians(i).sigma.apply(0,0))
      if ( gaussians(i).pdf(Vectors.dense(random)) == gaussianMixturePdf(gaussians, random) )
        generated = true
    }
    random
  }

  /**
    * Probability density function at given point corresponding to given gaussian mixture distribution
    * @param gaussians normal distributions representing a gaussian mixture, only one variable per gaussian is supported
    * @param x point whose probability to obtain
    * @return probablity ranging from 0 to 1
    */
  def gaussianMixturePdf(gaussians: List[MultivariateGaussian], x: Double): Double = {
    gaussians.map(g => g.pdf(Vectors.dense(x))).max
  }
}
