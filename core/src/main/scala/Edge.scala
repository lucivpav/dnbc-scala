import GaussianUtils.WeightedGaussian
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable.ListBuffer
import scala.util.Random

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
    * Call this when no more learning is to be done
    */
  def learnFinalize(): LearnedEdge[T]
}

/**
  * An edge that has been already trained and can now be used for inference
  * @tparam T type of states
  */
trait LearnedEdge[T] {
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
class DiscreteEdge extends Edge[String] with Serializable {
  override def learn(occurrence: String): Unit = {
      if ( counts.contains(occurrence) )
        counts += (occurrence -> (counts(occurrence)+1))
      else
        counts += (occurrence -> 1)
  }

  def learnFinalize(): LearnedDiscreteEdge = {
    val sum = counts.foldLeft(0)( (acc,kv) => acc + kv._2)
    for ( o <- counts ) {
      probabilities += (o._1 -> (o._2.toDouble / sum))
    }
    new LearnedDiscreteEdge(probabilities)
  }

  private var counts: Map[String,Int] = Map.empty
  private var probabilities: Map[String,Double] = Map.empty
}

/**
  * A trained discrete edge
  * @param probabilities describes probability function of give edge
  */
class LearnedDiscreteEdge(probabilities: Map[String,Double]) extends LearnedEdge[String] with Serializable {
  override def probability(state: String): Double = {
    if ( !probabilities.contains(state) )
      return 0
    probabilities(state)
  }
}

/**
  * The probabilities are expected to be continuous gaussian mixtures
  * @param k number of normal distributions in mixture
  */
class ContinuousEdge(sc: SparkContext, k: Int) {

  def learn(occurrence: Double): Unit = {
    occurrences += occurrence
  }

  def learnFinalize(): LearnedContinuousEdge = {
    val data = sc.parallelize(occurrences.map(o => Vectors.dense(o)))
    gaussianMixture = new GaussianMixture().setK(k).run(data)
    new LearnedContinuousEdge(gaussianMixture)
  }

  private var occurrences: ListBuffer[Double] = ListBuffer.empty
  private var gaussianMixture: GaussianMixtureModel = _
}

/**
  * Trained continuous edge
  * @param model describes underlying probability density funciton
  */
class LearnedContinuousEdge(model: GaussianMixtureModel) extends LearnedEdge[Double] with Serializable {
  override def probability(state: Double): Double = {
    val weightedGaussians = model.weights.indices.map(i => new WeightedGaussian(model.weights(i), model.gaussians(i)))
    GaussianUtils.gaussianMixturePdf(weightedGaussians.toList, state)
  }

  def getModel: GaussianMixtureModel = model
}

/**
  * Provides ability to generate random states given inner probability distribution
  * @tparam T type of states
  */
trait RandomEdge[T] {
  def next(): T
}

/**
  * @param probabilities describes probability function of give edge
  */
class RandomDiscreteEdge(probabilities: Map[String,Double]) extends RandomEdge[String] {
  override def next(): String = {
    val list = probabilities.toList
    list(RandomUtils.nextProbabilityIndex(list.map(p => p._2)))._1
  }
}

/**
  * @param gaussians normal distributions representing a gaussian mixture, only one variable per gaussian is supported
  */
class RandomContinuousEdge(gaussians: List[WeightedGaussian]) extends RandomEdge[Double] {

  /**
    * @return random number that corresponds to given gaussian mixture distribution
    */
  override def next(): Double = {
    GaussianUtils.nextGaussianMixture(gaussians)
  }
}
