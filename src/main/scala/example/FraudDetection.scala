package main.scala.example


import org.apache.spark._

import org.apache.spark.rdd.RDD
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.log4j.Logger


//LAB 10 TODO Import classes for MLLib regression labeledpoint, vectors, decisiontree, decisiontree model, MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object FraudDetection {

  case class CreditCardTransaction(fraud_ind:Double, amount: Double, distance: Double)
  val log = Logger.getLogger(getClass.getName)

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SparkFraudDetection").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    // function to parse input into CreditCardTransaction class
    def parseTransactions(str: String): CreditCardTransaction = {
      val line = str.split(",")
      CreditCardTransaction(line(0).toDouble, line(1).toDouble, line(2).toDouble)
    }

    /* -------------------------------- MLLIB------------------------------------------ */
    //Creating and RDD with the January 2014 data to be used for training the model
    val textRDD = sc.textFile("data/cross_val.csv")

    //val testDataRDD = sc.textFile("data/testdata.csv")

    val transactionsRDD = textRDD.map(parseTransactions).cache()

    //val testTransactionsRDD = testDataRDD.map(parseTransactions).cache()

     //- Defining the features array
    val mlprep = transactionsRDD.map(transaction => {
       val fraud_ind = transaction.fraud_ind
      val amount = transaction.amount
      val distance = transaction.distance
      Array(fraud_ind,amount,distance)
    })

   /* val testmlprep = testTransactionsRDD.map(transaction => {
      val fraud_ind = transaction.fraud_ind
      val amount = transaction.amount
      val distance = transaction.distance
      Array(fraud_ind,amount,distance)
    })*/

    //Making LabeledPoint of features - this is the training data for the model
    val mldata = mlprep.map(x => LabeledPoint(x(0), Vectors.dense(x(1), x(2))))


    mldata.take(10)

    //val testData = testmlprep.map(x => LabeledPoint(x(0), Vectors.dense(x(1), x(2))))

    val mldata0 = mldata.filter(x => x.label == 0).randomSplit(Array(0.85, 0.15))(1)

    println(s"count of non fraud transactions ="+mldata0.count())

    val mldata1 = mldata.filter(x => x.label != 0)

    println(s"count of  fraud transactions ="+mldata1.count())


    val mldata2 = mldata0 ++ mldata1

    val splits = mldata2.randomSplit(Array(0.9, 0.1))
    val (trainingData, testData) = (splits(0), splits(1))
    //val trainingData = splits(0)

    var categoricalFeaturesInfo = Map[Int, Int]()

    val numClasses = 2
    // Defning values for the other parameters
    val impurity = "gini"
    val maxDepth = 9
    val maxBins = 7000

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction,point.features)
    }
    println(s"------------------------------------------------------------------------------------")
    println(s"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Prediction Vs Actual  %%%%%%%%%%%%%%%%%%%")
    println(s"Actual  Prediction   Data---------------------------------------------------------------------------------------")
    labelAndPreds.toDF().show()


    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("total test data = "+testData.count())
    println("Test Error = " + testErr)


  }
}
