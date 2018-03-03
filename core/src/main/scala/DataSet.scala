import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Iterator for reading data from standardized data sets. See DataSetIterable for more details
  * @param types types of observed variables, can be "discrete" or "continuous"
  * @param lines an iterator reading data from data set
  */
class DataSetIterator(types: Array[String], lines: Iterator[String]) extends Iterator[Seq[State]] {
  private var finished = false

  override def hasNext: Boolean = !finished

  override def next(): Seq[State] = {
    val states = ListBuffer.empty[State]
    for ( line <- lines.takeWhile(l => l != ".") ) {
      if ( line == ".." ) {
        finished = true
        return states
      }
      val splitted = line.split(" ")
      val discreteValues = new ListBuffer[String]
      val continuousValues = new ListBuffer[Double]
      splitted.drop(1).zipWithIndex.foreach(z => types(z._2) match {
        case "discrete" => discreteValues += z._1
        case "continuous" => continuousValues += z._1.toDouble
      })
      states += new State(splitted(0), new ObservedState(discreteValues.toList, continuousValues.toList))
    }
    if ( !lines.hasNext )
      finished = true
    states
  }
}

/**
  * Helper class to use standardized data sets in DynamicNaiveBayesianClassifier
  * First line of data set must contain types of variables, possible types are: discrete, continuous
  * Sequences to be learned follow, separated by line "."
  * Training and testing sequences are separated by line ".."
  * See files in dataset folder for an example of a data set suitable for reading by this class
  * @param lines iterator of lines in data set file
  * @param learning learning data is being consumed if true, testing data otherwise
  */
class DataSetIterable(lines: Iterator[String], learning: Boolean) extends Iterable[Seq[State]] {
  private val types = getVariableTypes

  if ( !learning )
    lines.dropWhile(l => l != "..").drop(1)

  private def getVariableTypes: Array[String] = {
    val firstLine = lines.take(1).toList.head
    firstLine.split(" ").drop(1)
  }

  override def iterator: DataSetIterator = new DataSetIterator(types, lines)
}
