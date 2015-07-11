name := "SparkKmeansStudy"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.0",
  "org.apache.spark" %% "spark-mllib"  % "1.4.0",
  "org.apache.lucene" % "lucene-benchmark" % "3.5.0"
)