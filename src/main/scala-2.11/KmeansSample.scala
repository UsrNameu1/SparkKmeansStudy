/**
 * Created by yad on 15/07/05.
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object KmeansSample extends App {
  val context = new SparkContext("local", "demo")

  val data = context.
    textFile("src/main/resources/iris.data").
    filter(_.nonEmpty).
    map { s =>
    val elems = s.split(",")
    (elems.last, Vectors.dense(elems.init.map(_.toDouble)))
  }

  val k = 3 
  val maxItreations = 100
  val clusters = KMeans.train(data.map(_._2), k, maxItreations)

  clusters.clusterCenters.foreach {
    center => println(f"${center.toArray.mkString("[", ", ", "]")}%s")
  }

  data.foreach { tuple =>
    println(f"${tuple._2.toArray.mkString("[", ", ", "]")}%s " +
      f"(${tuple._1}%s) : cluster = ${clusters.predict(tuple._2)}%d")
  }
}
