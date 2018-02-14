import org.scalatest.{FunSuite, Ignore}

import scala.collection.mutable.ListBuffer
import scala.io.Source

// TODO: refactor: tests share similar functionality
class DynamicNaiveBayesianClassifierTest extends FunSuite {

  test("Single discrete observed variable") {
    var file = Source.fromFile("dataset/robot_no_momemtum.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[String]

    var successRate = ListBuffer.empty[Double]

    var hmm = new DynamicNaiveBayesianClassifier(200, 1, 0)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage) {
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedDiscreteVars += observedStates.toList
          hmm.learnSequence(hiddenStates.toList, observedDiscreteVars.toList, List.empty)
        }
        else
        {
          var observedDiscreteVars = ListBuffer.empty[List[String]]
          observedDiscreteVars += observedStates.toList
          val inferedHiddenStates = hmm.infereMostLikelyHiddenStates(observedDiscreteVars.toList, List.empty)
          if ( inferedHiddenStates.length != hiddenStates.length )
            throw new Exception("hidden states length mismatch")

          successRate += hiddenStates.zipWithIndex.count(o => o._1 == inferedHiddenStates(o._2)) / hiddenStates.length.toDouble
        }
        if (line == ".." ) {
          hmm.learnFinalize()
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

    var hmm = new DynamicNaiveBayesianClassifier(200, 0, 1)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage) {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          observedContinuousVars += observedStates.toList
          hmm.learnSequence(hiddenStates.toList, List.empty, observedContinuousVars.toList)
        }
        else
        {
          var observedContinuousVars = ListBuffer.empty[List[Double]]
          observedContinuousVars += observedStates.toList
          val inferedHiddenStates = hmm.infereMostLikelyHiddenStates(List.empty, observedContinuousVars.toList)
          if ( inferedHiddenStates.length != hiddenStates.length )
            throw new Exception("hidden states length mismatch")

          successRate += hiddenStates.zipWithIndex.count(o => o._1 == inferedHiddenStates(o._2)) / hiddenStates.length.toDouble
        }
        if (line == ".." ) {
          hmm.learnFinalize()
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
}
