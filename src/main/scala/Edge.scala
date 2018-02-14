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
