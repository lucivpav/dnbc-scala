import org.scalatest.FunSuite
import java.io.{File, PrintWriter}

class GaussianMixtureTest extends FunSuite {

  test("variable should have Gaussian mixture distribution") {
    var out = new PrintWriter(new File("random_gaussian_mixture.data"))
    val gaussians = List(new Gaussian(10, 25), new Gaussian(30, 16))
    val rg = new GaussianMixture(gaussians)
    for ( i <- 0 until 1000 )
      out.write(rg.nextRandom() + "\n")
    out.close()

    cancel("Plot histogram of the data and verify manually")
  }

  // TODO: test density
}
