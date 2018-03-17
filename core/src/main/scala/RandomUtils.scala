import scala.util.Random

object RandomUtils {
  /**
    *
    * @param probabilities assumed sum=1 and each p>=0
    * @return
    */
  def nextProbabilityIndex(probabilities: List[Double]): Int = {
    val random = Random.nextDouble()
    var acc = 0.0
    probabilities.zipWithIndex.foreach(z => {
      if ( acc < random && random <= acc+z._1 )
        return z._2
      acc += z._1
    })
    throw new Exception("RandomDiscreteEdge.next() failed")
  }
}
