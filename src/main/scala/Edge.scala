trait Edge[T] {
  def learn(occurence: T): Unit
  def learnFinalize(): Unit
  def probability(state: T): Double
}

class DiscreteEdge extends Edge[String] {
  override def learn(occurence: String): Unit = {
      if ( counts.contains(occurence) )
        counts += (occurence -> (counts(occurence)+1))
      else
        counts += (occurence -> 1)
  }

  override def learnFinalize(): Unit = {
    val sum = counts.foldLeft(0)( (acc,kv) => acc + kv._2)
    for ( o <- counts ) {
      probabilities += (o._1 -> (o._2.toDouble / sum))
    }
  }

  override def probability(state: String): Double = {
    if ( !probabilities.contains(state) )
      return 0
    probabilities(state)
  }

  private var counts: Map[String,Int] = Map.empty
  private var probabilities: Map[String,Double] = Map.empty
}


import scala.collection.mutable.ListBuffer

class ContinuousEdge() extends Edge[Double] {

  override def learn(occurence: Double): Unit = {
    occurrences += occurence
  }

  override def learnFinalize(): Unit = {
    mean = occurrences.sum / occurrences.length
    variance = occurrences.map(o => scala.math.pow(o-mean,2)).sum / occurrences.length
  }

  override def probability(state: Double): Double = {
    (1/scala.math.sqrt(2*scala.math.Pi*variance))*scala.math.pow(scala.math.E,-scala.math.pow(state - mean, 2)/(2*variance)) //eh.
  }

  private var occurrences: ListBuffer[Double] = ListBuffer.empty
  private var mean: Double = 0.0
  private var variance: Double = 0.0
}
