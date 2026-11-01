name := "tscm-listener"
version := "0.1.0"
scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.1" % Provided,
  "org.apache.spark" %% "spark-sql" % "3.5.1" % Provided
)
