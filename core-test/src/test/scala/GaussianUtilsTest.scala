import org.scalatest.FunSuite
import java.io.{File, PrintWriter}

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
    var modelOut = new PrintWriter(new File("random_gaussian_mixture.model.data"))
    val gaussians = List(new MultivariateGaussian(Vectors.dense(10), Matrices.dense(1,1, Array(25))),
                          new MultivariateGaussian(Vectors.dense(30), Matrices.dense(1,1, Array(16))))
    var edge = new ContinuousEdge(TestUtils.GetSparkContext(), 2)
    for ( i <- 0 until 1000 ) {
      val random = GaussianUtils.nextGaussianMixture(gaussians)
      edge.learn(random)
      out.write(random + "\n")
    }
    val learnedEdge = edge.learnFinalize()
    val model = learnedEdge.getModel
    modelOut.write("weight mean variance\n")
    (0 until 2).foreach( i => {
      modelOut.write(model.weights(i).toString + " "
                      + model.gaussians(i).mu(0) + " "
                      + model.gaussians(i).sigma.apply(0,0)
                      + "\n")
    })

    modelOut.close()
    out.close()

    cancel("Plot histogram of the data and verify manually")
  }

  // TODO: test mixture pdf
}
