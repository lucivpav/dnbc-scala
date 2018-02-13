import scala.collection.mutable.ListBuffer
import scala.io.Source

object Main {
  def getSuccessRate(expectedHiddenStates: List[String], inferedHiddenStates: List[String]): Double = {
    var successCount = 0
    inferedHiddenStates.zipWithIndex.takeWhile{case(state,idx) => expectedHiddenStates(idx) == state}.foreach
    {_ => successCount += 1}
    successCount.toDouble / inferedHiddenStates.length
  }

  def main(args: Array[String]): Unit = {

    var file = Source.fromFile("dataset/robot_no_momemtum.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[String]

    var successRate = ListBuffer.empty[Double]
    var successRateWithoutInitialState = ListBuffer.empty[Double]

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
          val inferedHiddenStates = hmm.infereMostLikelyHiddenStates(observedStates.toList)
          if ( inferedHiddenStates.length != hiddenStates.length )
            throw new Exception("hidden states length mismatch")

          successRate += getSuccessRate(hiddenStates.toList, inferedHiddenStates)
          successRateWithoutInitialState += getSuccessRate(hiddenStates.toList.slice(1,hiddenStates.length),
                                                            inferedHiddenStates.slice(1,inferedHiddenStates.length))
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
    val avg1 = (successRate.sum / successRate.length) * 100
    val avg2 = (successRateWithoutInitialState.sum / successRateWithoutInitialState.length) * 100
    println(s"Average succcess rate: $avg1%")
    println(s"Average success rate after correct first step: $avg2%")
  }
}
