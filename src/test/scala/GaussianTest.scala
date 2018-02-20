import org.scalatest.FunSuite
import java.io.{File, PrintWriter}

class GaussianTest extends FunSuite {

  test("variable should have Gaussian distribution") {
    var out = new PrintWriter(new File("random_gaussian.data"))
    val rg = new Gaussian(10, 5)
    for ( i <- 0 until 1000 )
      out.write(rg.nextRandom() + "\n")
    out.close()
    cancel("Plot histogram of the data and verify manually")
  }

  // TODO: test density
}
