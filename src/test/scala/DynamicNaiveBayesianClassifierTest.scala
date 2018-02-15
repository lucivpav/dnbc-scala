import org.scalatest.FunSuite

import scala.collection.mutable.ListBuffer
import scala.io.Source

// TODO: refactor: tests share similar functionality
class DynamicNaiveBayesianClassifierTest extends FunSuite {

  test("Single discrete observed variable") {
    var file = Source.fromFile("dataset/robot_no_momemtum.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[String]

    var successRate = ListBuffer.empty[Double]

    var dnbc = new DynamicNaiveBayesianClassifier(200, 1, 0)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage) {
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedDiscreteVars += observedStates.toList
          dnbc.learnSequence(hiddenStates.toList, observedDiscreteVars.toList, List.empty)
        }
        else
        {
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedDiscreteVars += observedStates.toList
          val inferedHiddenStates = dnbc.infereMostLikelyHiddenStates(observedDiscreteVars.toList, List.empty)
          if ( inferedHiddenStates.lengthCompare(hiddenStates.length) != 0 )
            throw new Exception("hidden states length mismatch")

          successRate += hiddenStates.zipWithIndex.count(o => o._1 == inferedHiddenStates(o._2)) / hiddenStates.length.toDouble
        }
        if (line == ".." ) {
          dnbc.learnFinalize()
          learningStage = false
        }
        hiddenStates = ListBuffer.empty[String]
        observedStates = ListBuffer.empty[String]
      }
      else
      {
        hiddenStates += line.substring(0, 3)
        observedStates += line.substring(4)
      }
    }
    successRate.foreach(r => println((r * 100).toString + "%"))
    val avg = (successRate.sum / successRate.length) * 100
    println(s"Average success rate: $avg%")

    assert( avg > 65 )
  }

  test("Single continuous observed variable") {
    var file = Source.fromFile("dataset/robot_no_momemtum_continuous.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[Double]

    var successRate = ListBuffer.empty[Double]

    var dnbc = new DynamicNaiveBayesianClassifier(200, 0, 1)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage) {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          observedContinuousVars += observedStates.toList
          dnbc.learnSequence(hiddenStates.toList, List.empty, observedContinuousVars.toList)
        }
        else
        {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          observedContinuousVars += observedStates.toList
          val inferedHiddenStates = dnbc.infereMostLikelyHiddenStates(List.empty, observedContinuousVars.toList)
          if ( inferedHiddenStates.lengthCompare(hiddenStates.length) != 0 )
            throw new Exception("hidden states length mismatch")

          successRate += hiddenStates.zipWithIndex.count(o => o._1 == inferedHiddenStates(o._2)) / hiddenStates.length.toDouble
        }
        if (line == ".." ) {
          dnbc.learnFinalize()
          learningStage = false
        }
        hiddenStates = ListBuffer.empty[String]
        observedStates = ListBuffer.empty[Double]
      }
      else
      {
        hiddenStates += line.substring(0, 3)
        observedStates += line.substring(4).toDouble
      }
    }
    successRate.foreach(r => println((r * 100).toString + "%"))
    val avg = (successRate.sum / successRate.length) * 100
    println(s"Average success rate: $avg%")

    assert( avg > 40 )
  }

  test("One continuous and one discrete observed variable") {
    var file = Source.fromFile("dataset/robot_no_momemtum_bivariate.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedContinuousStates = ListBuffer.empty[Double]
    var observedDiscreteStates = ListBuffer.empty[String]

    var successRate = ListBuffer.empty[Double]

    var dnbc = new DynamicNaiveBayesianClassifier(200, 1, 1)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage) {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedContinuousVars += observedContinuousStates.toList
          observedDiscreteVars += observedDiscreteStates.toList
          dnbc.learnSequence(hiddenStates.toList, observedDiscreteVars.toList, observedContinuousVars.toList)
        }
        else
        {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedContinuousVars += observedContinuousStates.toList
          observedDiscreteVars += observedDiscreteStates.toList
          val inferedHiddenStates = dnbc.infereMostLikelyHiddenStates(observedDiscreteVars.toList, observedContinuousVars.toList)
          if ( inferedHiddenStates.lengthCompare(hiddenStates.length) != 0 )
            throw new Exception("hidden states length mismatch")

          successRate += hiddenStates.zipWithIndex.count(o => o._1 == inferedHiddenStates(o._2)) / hiddenStates.length.toDouble
        }
        if (line == ".." ) {
          dnbc.learnFinalize()
          learningStage = false
        }
        hiddenStates = ListBuffer.empty[String]
        observedContinuousStates = ListBuffer.empty[Double]
        observedDiscreteStates = ListBuffer.empty[String]
      }
      else
      {
        val splitted = line.split(" ")
        hiddenStates += splitted(0)
        observedContinuousStates += splitted(1).toDouble
        observedDiscreteStates += splitted(2)
      }
    }
    successRate.foreach(r => println((r * 100).toString + "%"))
    val avg = (successRate.sum / successRate.length) * 100
    println(s"Average success rate: $avg%")

    assert( avg > 55 )
  }

  test("Passing invalid parameters should result in an exception") {
    var dnbc = new DynamicNaiveBayesianClassifier(2, 1, 1)

    val hiddenStates = Seq("A", "B").toList
    val discreteVariables = Seq(Seq("r", "b").toList).toList
    val continuousVariables = Seq(Seq(42.666, 1.1).toList).toList

    val invalidHiddenStates = Seq("A").toList
    val invalidDiscreteVariables = Seq(Seq("r").toList).toList
    val invalidContinuousVariables = Seq(Seq(42.666).toList).toList

    intercept[Exception] { dnbc.learnSequence(hiddenStates, invalidDiscreteVariables, invalidContinuousVariables) }
    intercept[Exception] { dnbc.learnSequence(hiddenStates, discreteVariables, invalidContinuousVariables) }
    intercept[Exception] { dnbc.learnSequence(hiddenStates, invalidDiscreteVariables, continuousVariables) }
    intercept[Exception] { dnbc.learnSequence(invalidHiddenStates, discreteVariables, continuousVariables) }
    intercept[Exception] { dnbc.learnSequence(hiddenStates, invalidDiscreteVariables, invalidContinuousVariables) }
    dnbc.learnSequence(hiddenStates, discreteVariables, continuousVariables)
    dnbc.learnFinalize()
    intercept[Exception] { dnbc.infereMostLikelyHiddenStates(invalidDiscreteVariables, invalidContinuousVariables) }
    intercept[Exception] { dnbc.infereMostLikelyHiddenStates(discreteVariables, invalidContinuousVariables) }
    intercept[Exception] { dnbc.infereMostLikelyHiddenStates(invalidDiscreteVariables, continuousVariables) }
    dnbc.infereMostLikelyHiddenStates(discreteVariables, continuousVariables)
  }
}
