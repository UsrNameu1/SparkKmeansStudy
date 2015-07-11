/**
 * Created by yad on 15/07/05.
 */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.slf4j.Logger

object KmeansSample extends App {
  val context = new SparkContext("local", "reuter-demo")

  val documents = context
    .wholeTextFiles("src/main/resources/reuter/extracted/")
    .map{ _._2 }
//    .map{ text => text.lines }
//    .map{ lines => lines.drop(lines.size - 2).next() }
    .map{ _.split(" ").toSeq }

  val hashingTF = new HashingTF()
  val tf = hashingTF.transform(documents)

  tf.cache()
  val idf = new IDF().fit(tf)
  val tfidf = idf.transform(tf)

  val k = 20
  val maxItreations = 50
  val clusters = KMeans.train(tfidf, k, maxItreations)

//  clusters.clusterCenters.foreach {
//    center => println(f"${center.toArray.mkString("[", ", ", "]")}%s")
//  }
//
//  tfidf.foreach { tuple =>
//    println(f"${tuple._2.toArray.mkString("[", ", ", "]")}%s " +
//      f"(${tuple._1}%s) : cluster = ${clusters.predict(tuple._2)}%d")
//  }
}
