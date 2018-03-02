name := "dnbc-scala"

version := "0.1"

val sparkVer = "2.1.0"

val commonSettings = Seq(scalaVersion := "2.11.8", libraryDependencies := Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer,
    "org.scalatest" %% "scalatest" % sparkVer
))

lazy val preprocessing = project in file("preprocessing") dependsOn core settings commonSettings
lazy val core = project in file ("core") settings commonSettings
