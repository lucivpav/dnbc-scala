import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

import scala.io.Source
import java.io.File
import java.util.concurrent.TimeUnit

class DynamicNaiveBayesianClassifierTest extends FunSuite {

  private val sc = SparkSession.builder.appName("Simple Application").config("spark.master", "local")
                                .getOrCreate().sparkContext
  sc.setLogLevel("ERROR")

  test("Single discrete observed variable") {
    val avg = measurePerformance("/robot_no_momentum.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 65 )
  }

  test("Single continuous observed variable") {
    val avg = measurePerformance("/robot_no_momentum_continuous.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 40 )
  }

  test("One continuous and one discrete observed variable") {
    val avg = measurePerformance("/robot_no_momentum_bivariate.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 55 )
  }

  // warning: this test may sometimes fail, due to the nature of GM, whose success depends on luck with initial guess
  test("Variable with Gaussian mixture") {
    val dataSetPath = "/gaussian_mixture.data"
    val rate1 = measurePerformance(dataSetPath, Option(List(1))).successRate
    val rate2 = measurePerformance(dataSetPath, Option(List(2))).successRate
    assert ( rate1+0.03 < rate2 )
  }

  test("Passing invalid parameters should result in an exception") {
    val validObservedState = new ObservedState(List("b"), List(1.1))
    val invalidObservedState = new ObservedState(List("b", "c"), List(1.1))

    val firstState = new State("A", new ObservedState(List("r"), List(42.666)))
    val secondState = new State("B", validObservedState)
    val invalidSecondState = new State("B", invalidObservedState)

    val validSequence = Seq(firstState, secondState)
    val invalidSequence = Seq(firstState, invalidSecondState)

    intercept[Exception] { DynamicNaiveBayesianClassifier.mle(sc, Seq(validSequence), Option(List(0))) }
    intercept[Exception] { DynamicNaiveBayesianClassifier.mle(sc, Seq(validSequence), Option(List(1, 2))) }
    intercept[Exception] { DynamicNaiveBayesianClassifier.mle(sc, Seq(invalidSequence)) }
    val model = DynamicNaiveBayesianClassifier.mle(sc, Seq(validSequence))
    intercept[Exception] { model.inferMostLikelyHiddenStates(Seq(validObservedState, invalidObservedState)) }
    model.inferMostLikelyHiddenStates(Seq(validObservedState, validObservedState))
  }

  test("Assure sequential implementation computes big data set in reasonable time") {
    val dataSetPath = "dataset/performance.data"
    assert ( new File(dataSetPath).exists )
    val perf = measurePerformance(dataSetPath, Option.empty, false)
    println(s"Average success rate: ${perf.successRate}%")
    assert( perf.successRate > 10 )
    val learningTime = TimeUnit.SECONDS.convert(perf.learningTime, TimeUnit.NANOSECONDS)
    val testingTime = TimeUnit.SECONDS.convert(perf.testingTime, TimeUnit.NANOSECONDS)
    println(s"Learning time: $learningTime\n Testing time: $testingTime")
    assert( learningTime > 60 && learningTime < 4*60 )
    assert( testingTime > 45 && learningTime < 4*45 )
  }

  /**
    * Describes performance on given data set
    * @param successRate average success rate on provided data set
    * @param learningTime time spent learning in nanoseconds
    * @param testingTime time spent on inference in nanoseconds
    */
  case class Performance(successRate: Double, learningTime: Long, testingTime: Long)

  /**
    * Measures performance of DynamicNaiveBayesianClassifier on standardized data set
    * @param dataSetPath path to data set
    * @param hints number of normal distributions in each continuous variable
    * @param resourcesPath indicates whether the data set is stored in resources folder
    * @return performance on given data set
    */
  private def measurePerformance(dataSetPath: String,
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
      secondReader= Source.fromFile(dataSetPath).getLines()
    }

    val learningIterable = new DataSetIterable(firstReader, true)
    val testingIterable = new DataSetIterable(secondReader, false)

    val learningTimeBegin = System.nanoTime()
    val model = DynamicNaiveBayesianClassifier.mle(sc, learningIterable, hints)
    val learningTimeDuration = System.nanoTime()-learningTimeBegin

    val batches = testingIterable.map(seq => {
      val testingTimeBegin = System.nanoTime()
      val inferredHiddenStates = model.inferMostLikelyHiddenStates(seq.map(s => s.ObservedState))
      val testingTimeDuration = (System.nanoTime()-testingTimeBegin)

      val correctCount = seq.map(s => s.HiddenState).zipWithIndex.count(z => z._1 == inferredHiddenStates(z._2))
      Performance(correctCount / inferredHiddenStates.length.toDouble, -1, testingTimeDuration)
    }).toList
    val avg = (batches.map(p => p.successRate).sum / batches.length) * 100
    val testingTimeDuration = batches.map(p => p.testingTime).sum
    Performance(avg, learningTimeDuration, testingTimeDuration)
  }
}
