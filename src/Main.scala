import scala.collection.mutable.ListBuffer
import scala.io.Source

object Main {
  def main(args: Array[String]): Unit = {

    var file = Source.fromFile("dataset/robot_no_momemtum.data")
    var hiddenStates = ListBuffer.empty[String]
    var observedStates = ListBuffer.empty[String]

    var hmm = new HiddenMarkovModel(200)
    val trainingSize = 100
    var sequenceNumber = 1

    for ( line <- file.getLines() if sequenceNumber < trainingSize )
    {
      if ( line == "." )
      {
        hmm.learnSequence(hiddenStates.toList, observedStates.toList)
        hiddenStates = ListBuffer.empty[String]
        observedStates = ListBuffer.empty[String]
        sequenceNumber += 1
      }
      else
      {
        hiddenStates += line.substring(0, 3)
        observedStates += line.substring(4)
      }
    }

    hmm.learnFinalize()
    println("learning finished")
  }
}
