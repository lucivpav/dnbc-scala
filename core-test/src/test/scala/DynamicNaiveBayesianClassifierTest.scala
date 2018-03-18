import org.scalatest.FunSuite

class DynamicNaiveBayesianClassifierTest extends FunSuite {
  private val sc = TestUtils.GetSparkContext()

  test("Single discrete observed variable") {
    val avg = Performance.Measure(sc, "/robot_no_momentum.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 65 )
  }

  test("Single continuous observed variable") {
    val avg = Performance.Measure(sc, "/robot_no_momentum_continuous.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 40 )
  }

  test("One continuous and one discrete observed variable") {
    val avg = Performance.Measure(sc, "/robot_no_momentum_bivariate.data").successRate
    println(s"Average success rate: $avg%")
    assert( avg > 72 )
  }

  // warning: this test may sometimes fail, due to the nature of GM, whose success depends on luck with initial guess
  test("Variable with Gaussian mixture") {
    val dataSetPath = "/gaussian_mixture.data"
    val rate1 = Performance.Measure(sc, dataSetPath, Option(List(1))).successRate
    val rate2 = Performance.Measure(sc, dataSetPath, Option(List(2))).successRate
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
}
