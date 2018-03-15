import org.apache.spark.SparkContext

import scala.io.Source

/**
  * Describes performance on given data set
  * @param successRate average success rate on provided data set
  * @param learningTime time spent learning in nanoseconds
  * @param testingTime time spent on inference in nanoseconds
  */
case class Performance(successRate: Double, learningTime: Long, testingTime: Long)

object Performance {
  /**
    * Measures performance of DynamicNaiveBayesianClassifier on standardized data set
    *
    * @param dataSetPath   path to data set
    * @param hints         number of normal distributions in each continuous variable
    * @param resourcesPath indicates whether the data set is stored in resources folder
    * @return performance on given data set
    */
  def Measure(sparkContext: SparkContext,
              mle: Boolean,
              dataSetPath: String,
              hints: Option[List[Int]] = Option.empty,
              resourcesPath: Boolean = true): Performance = {
    var firstReader: Iterator[String] = null
    var secondReader: Iterator[String] = null

    if (resourcesPath) {
      firstReader = Source.fromInputStream(getClass.getResourceAsStream(dataSetPath)).getLines()
      secondReader = Source.fromInputStream(getClass.getResourceAsStream(dataSetPath)).getLines()
    }
    else {
      firstReader = Source.fromFile(dataSetPath).getLines()
      secondReader = Source.fromFile(dataSetPath).getLines()
    }

    val learningIterable = new DataSetIterable(firstReader, true)
    val testingIterable = new DataSetIterable(secondReader, false)

    val learningTimeBegin = System.nanoTime()
    val model = if (mle)
                  DynamicNaiveBayesianClassifier.mle(sparkContext, learningIterable, hints)
                else
                  DynamicNaiveBayesianClassifier.baumWelch(learningIterable)

    val learningTimeDuration = System.nanoTime() - learningTimeBegin

    val batches = testingIterable.map(seq => {
      val testingTimeBegin = System.nanoTime()
      val inferredHiddenStates = model.inferMostLikelyHiddenStates(seq.map(s => s.ObservedState))
      val testingTimeDuration = System.nanoTime() - testingTimeBegin

      val correctCount = seq.map(s => s.HiddenState).zipWithIndex.count(z => z._1 == inferredHiddenStates(z._2))
      Performance(correctCount / inferredHiddenStates.length.toDouble, -1, testingTimeDuration)
    }).toList
    val avg = (batches.map(p => p.successRate).sum / batches.length) * 100
    val testingTimeDuration = batches.map(p => p.testingTime).sum
    Performance(avg, learningTimeDuration, testingTimeDuration)
  }
}
