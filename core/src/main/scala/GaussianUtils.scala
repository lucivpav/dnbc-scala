import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

import scala.util.Random

object GaussianUtils {

  def nextGaussian(mean: Double, variance: Double): Double = {
    Random.nextGaussian()*scala.math.sqrt(variance)+mean
  }

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

  def gaussianMixturePdf(gaussians: List[MultivariateGaussian], x: Double): Double = {
    gaussians.map(g => g.pdf(Vectors.dense(x))).max
  }
}
