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

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}

import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.Vectors

// k: number of normal distributions in a mixture
class ContinuousEdge(sc: SparkContext, k: Int) extends Edge[Double] {

  override def learn(occurence: Double): Unit = {
    occurrences += occurence
  }

  override def learnFinalize(): Unit = {
    val data = sc.parallelize(occurrences.map(o => Vectors.dense(o)))
    gaussianMixture = new GaussianMixture().setK(k).run(data)
  }

  override def probability(state: Double): Double = {
    GaussianUtils.gaussianMixturePdf(gaussianMixture.gaussians.toList, state)
  }

  private var occurrences: ListBuffer[Double] = ListBuffer.empty
  private var gaussianMixture: GaussianMixtureModel = _
}
