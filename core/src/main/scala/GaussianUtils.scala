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

  class WeightedGaussian(weight: Double, gaussian: MultivariateGaussian) {
    def Weight: Double = weight
    def Gaussian: MultivariateGaussian = gaussian
  }


  /**
    * Returns a randomly generated number from gaussian mixture distribution
    * @param gaussians normal distributions representing a gaussian mixture, only one variable per gaussian is supported
    * @return random number that corresponds to given gaussian mixture distribution
    */
  def nextGaussianMixture(gaussians: List[WeightedGaussian]): Double = {
    /* TODO: extract. this shares similar functionality with RandomDiscreteEdge */
    val random = Random.nextDouble()
    var acc = 0.0
    var idx = -1
    gaussians.zipWithIndex.foreach(z => {
      if ( acc < random && random <= acc+z._1.Weight )
        idx = z._2
      acc += z._1.Weight
    })
    if (idx == -1)
      throw new Exception("nextGaussianMixture() failed")

    nextGaussian(gaussians(idx).Gaussian.mu(0), gaussians(idx).Gaussian.sigma.apply(0,0))
  }

  /**
    * Probability density function at given point corresponding to given gaussian mixture distribution
    * @param gaussians normal distributions representing a gaussian mixture, only one variable per gaussian is supported
    * @param x point whose probability to obtain
    * @return probablity ranging from 0 to 1
    */
  def gaussianMixturePdf(gaussians: List[WeightedGaussian], x: Double): Double = {
    gaussians.map(g => g.Weight * g.Gaussian.pdf(Vectors.dense(x))).sum
  }
}
