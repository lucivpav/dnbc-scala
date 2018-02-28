import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class DynamicNaiveBayesianClassifierTest extends FunSuite {

  private val sc = SparkSession.builder.appName("Simple Application").config("spark.master", "local")
                                .getOrCreate().sparkContext

  test("Single discrete observed variable") {
    val avg = measurePerformance("dataset/robot_no_momentum.data")
    println(s"Average success rate: $avg%")
    assert( avg > 65 )
  }

  test("Single continuous observed variable") {
    val avg = measurePerformance("dataset/robot_no_momentum_continuous.data")
    println(s"Average success rate: $avg%")
    assert( avg > 40 )
  }

  test("One continuous and one discrete observed variable") {
    val avg = measurePerformance("dataset/robot_no_momentum_bivariate.data")
    println(s"Average success rate: $avg%")
    assert( avg > 55 )
  }

  // warning: this test may sometimes fail, due to the nature of GM, whose success depends on luck with initial guess
  test("Variable with Gaussian mixture") {
    val dataSetPath = "dataset/gaussian_mixture.data"
    val rate1 = measurePerformance(dataSetPath, Option(List(1)))
    val rate2 = measurePerformance(dataSetPath, Option(List(2)))
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

  // returns average success rate on provided data set
  // hints: number of gaussians in each continuos variable
  private def measurePerformance(dataSetPath: String, hints: Option[List[Int]] = Option.empty): Double = {
    val learningIterable = new DataSetIterable(dataSetPath, true)
    val testingIterable = new DataSetIterable(dataSetPath, false)
    val model = DynamicNaiveBayesianClassifier.mle(sc, learningIterable, hints)
    val successRates = testingIterable.map(seq => {
      val inferredHiddenStates = model.inferMostLikelyHiddenStates(seq.map(s => s.ObservedState))
      val correctCount = seq.map(s => s.HiddenState).zipWithIndex.count(z => z._1 == inferredHiddenStates(z._2))
      correctCount / inferredHiddenStates.length.toDouble
    }).toList
    (successRates.sum / successRates.length) * 100
  }
}
