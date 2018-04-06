name := "dnbc-scala"

version := "0.1"

val sparkVer = "2.1.0"

enablePlugins(PackPlugin)
packMain := Map("hello" -> "org.mydomain.Hello")

val commonSettings = Seq(scalaVersion := "2.11.8", fork := true, libraryDependencies := Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer,
    "org.scalatest" %% "scalatest" % sparkVer
))

lazy val preprocessing = project in file("preprocessing") settings commonSettings dependsOn(core, testUtils)
lazy val core = project in file ("core") settings commonSettings
lazy val coreTest = project in file("core-test") settings commonSettings dependsOn(core, testUtils)
lazy val testUtils = project in file("test-utils") settings commonSettings dependsOn core
lazy val performance = project in file("performance") settings commonSettings dependsOn testUtils
