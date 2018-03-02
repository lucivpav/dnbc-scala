/**
  * Probabilities of transitions to states
  * @tparam T type of states
  */
trait Edge[T] {
  /**
    * Notifies an Edge that there is a transition to state in occurrence
    * @param occurrence state that has occurred
    */
  def learn(occurrence: T): Unit

  /**
    * Call this when no more learning is to be done. This will allow one to call probability()
    */
  def learnFinalize(): Unit

  /**
    * A probability of transitioning to given state
    * @param state state to transition into
    * @return a real number from zero to one
    */
  def probability(state: T): Double
}

/**
  * The probabilities are expected to be discrete
  */
class DiscreteEdge extends Edge[String] {
  override def learn(occurrence: String): Unit = {
      if ( counts.contains(occurrence) )
        counts += (occurrence -> (counts(occurrence)+1))
      else
        counts += (occurrence -> 1)
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

/**
  * The probabilities are expected to be continuous gaussian mixtures
  * @param sc spark context
  * @param k number of normal distributions in mixture
  */
class ContinuousEdge(sc: SparkContext, k: Int) extends Edge[Double] {

  override def learn(occurrence: Double): Unit = {
    occurrences += occurrence
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
