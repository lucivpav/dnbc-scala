import scala.util.Random

class GaussianMixture(gaussians: List[Gaussian]) {
  def nextRandom(): Double = {
    var generated = false
    var random = 0.0
    while (!generated) {
      val i = Random.nextInt(gaussians.length)
      random = gaussians(i).nextRandom()
      if ( gaussians(i).density(random) == density(random) )
        generated = true
    }
    random
  }

  def density(x: Double): Double = {
    gaussians.map(g => g.density(x)).max
  }
}
