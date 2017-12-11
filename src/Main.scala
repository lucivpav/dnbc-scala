import scala.collection.mutable.ListBuffer
import scala.io.Source

object Main {
  def main(args: Array[String]): Unit = {

    var file = Source.fromFile("dataset/robot_no_momemtum.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[String]

    var hmm = new HiddenMarkovModel(200)
    var learningStage = true

    for ( line <- file.getLines() )
    {
      if ( line == "." || line == ".." ){
        if (learningStage)
          hmm.learnSequence(hiddenStates.toList, observedStates.toList)
        else
          hmm.infereMostLikelyHiddenStates(observedStates.toList)
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

  }
}
