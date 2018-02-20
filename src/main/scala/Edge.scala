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

import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.Vectors

abstract class ContinuousEdge() extends Edge[Double]

class ContinuousGaussianEdge() extends ContinuousEdge {

  override def learn(occurence: Double): Unit = {
    occurrences += occurence
  }

  override def learnFinalize(): Unit = {
    val mean = occurrences.sum / occurrences.length
    val variance = occurrences.map(o => scala.math.pow(o-mean,2)).sum / occurrences.length
    gaussian = new Gaussian(mean, variance)
  }

  override def probability(state: Double): Double = {
    gaussian.density(state)
  }

  private var occurrences: ListBuffer[Double] = ListBuffer.empty
  private var gaussian: Gaussian = _
}

// k: number of normal distributions in a mixture
class ContinuousGaussianMixtureEdge(sc: SparkContext, k: Int) extends ContinuousEdge {

  override def learn(occurence: Double): Unit = {
    occurrences += occurence
  }

  override def learnFinalize(): Unit = {
    val data = sc.parallelize(occurrences.map(o => Vectors.dense(o)))
    val gmm = new org.apache.spark.mllib.clustering.GaussianMixture().setK(k).run(data)
    val gaussians = ListBuffer.empty[Gaussian]
    for ( g <- gmm.gaussians )
      gaussians += new Gaussian(g.mu(0), g.sigma.apply(0,0))
    gaussianMixture = new GaussianMixture(gaussians.toList)
  }

  override def probability(state: Double): Double = {
    gaussianMixture.density(state)
  }

  private var occurrences: ListBuffer[Double] = ListBuffer.empty
  private var gaussianMixture: GaussianMixture = _
}
