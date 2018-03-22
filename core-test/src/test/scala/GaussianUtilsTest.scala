import org.scalatest.FunSuite
import java.io.{File, PrintWriter}

import GaussianUtils.WeightedGaussian
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

class GaussianUtilsTest extends FunSuite {

  test("Variable should have Gaussian distribution") {
    var out = new PrintWriter(new File("random_gaussian.data"))
    for ( i <- 0 until 1000 )
      out.write(GaussianUtils.nextGaussian(10,5) + "\n")
    out.close()
    cancel("Plot histogram of the data and verify manually")
  }

  test("Variable should have Gaussian mixture distribution") {
    var out = new PrintWriter(new File("random_gaussian_mixture.data"))
    val gaussians = List(new WeightedGaussian(0.5, new MultivariateGaussian(Vectors.dense(10),
                                                                            Matrices.dense(1,1, Array(25)))),
                          new WeightedGaussian(0.5, new MultivariateGaussian(Vectors.dense(30),
                                                                              Matrices.dense(1,1, Array(16)))))
    for ( i <- 0 until 1000 )
      out.write(GaussianUtils.nextGaussianMixture(gaussians) + "\n")
    out.close()

    cancel("Plot histogram of the data and verify manually")
  }

  // TODO: test mixture pdf
}
