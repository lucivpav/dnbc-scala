import scala.collection.mutable.ListBuffer
import scala.io.Source

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

class DataSetIterable(fileName: String, learning: Boolean) extends Iterable[Seq[State]] {
  private val file = Source.fromFile(fileName)
  private val lines = file.getLines()
  private val types = getVariableTypes

  if ( !learning )
    lines.dropWhile(l => l != "..").drop(1)

  private def getVariableTypes: Array[String] = {
    val firstLine = lines.take(1).toList.head
    firstLine.split(" ").drop(1)
  }

  override def iterator: DataSetIterator = new DataSetIterator(types, lines)
}
