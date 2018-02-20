name := "dnbc-scala"

version := "0.1"

scalaVersion := "2.10.7"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "2.0.0",
  "org.apache.spark" % "spark-mllib_2.10" % "2.0.0",
  "org.scalatest" % "scalatest_2.10" % "2.0.0" % "test"
)