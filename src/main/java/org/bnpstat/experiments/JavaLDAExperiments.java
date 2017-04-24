package org.bnpstat.experiments;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.*;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;

import java.io.*;
import java.util.ArrayList;
import java.util.Properties;

/**
 * Created by hvhuynh on 2/11/2016.
 */
public class JavaLDAExperiments {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "V:\\Code\\Java\\BNPStat\\hadoop-common-2.2.0-bin-master\\");

        String currentDir = null;
        String propFileName = null;
        if (args.length == 0) {
            propFileName = "config.properties";
            System.out.println("Using the default setting file. Pass the configure file to use the user's config");
        } else if (args.length == 1) {
            propFileName = args[0];
        } else {
            System.out.println("Usage:java <classname>  or \n java <classname>  <configuration file> \n Please check \'config.properties\' file for template");
            System.exit(0);
        }

        Properties prop = new Properties();
        InputStream input = null;
        try {
            currentDir = new java.io.File(".").getCanonicalPath();

            //input = new FileInputStream(propFileName);

            // input=MC2ExperimentNIPSAuthorContext.class.getClassLoader().getResourceAsStream(propFileName);
            System.out.println(propFileName);
            input = new FileInputStream(propFileName);

            // load a properties file
            prop.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }


        String contFilePath = prop.getProperty("lda.contentPath");
        String outFolderPath = currentDir + File.separator + prop.getProperty("lda.outFolderPath");
        String testFilePath = currentDir + File.separator + prop.getProperty("lda.testPath");

        System.out.println("Current folder" + outFolderPath);

        SparkConf conf = new SparkConf().setAppName("LDA Example").setMaster("local").set("spark.driver.maxResultSize", "2g");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load and parse the data
//        String contFilePath = "data/pubmed/2000now/pubmed_content_small.txt";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), contFilePath).toJavaRDD();

        //JavaRDD<String> data = sc.textFile(path);
//        JavaRDD<Vector> parsedData = data.map(
//                new Function<LabeledPoint, Vector>() {
//                    public Vector call(String s) {
//                        String[] sarray = s.trim().split(" ");
//                        double[] values = new double[sarray.length];
//                        for (int i = 0; i < sarray.length; i++)
//                            values[i] = Double.parseDouble(sarray[i]);
//                        return Vectors.dense(values);
//                    }
//                }
//        );
        // Index documents with unique IDs
        JavaPairRDD<Long, Vector> corpus = data.mapToPair(
                new PairFunction<LabeledPoint, Long, Vector>() {

                    public Tuple2<Long, Vector> call(LabeledPoint point) {
                        return new Tuple2((long) point.label(), point.features());
                    }
                }
        );
        corpus.cache();

        // Cluster the documents into three topics using LDA
        OnlineLDAOptimizer optimizer = new OnlineLDAOptimizer();
        optimizer.setKappa(0.9);
        optimizer.setTau0(1);
        optimizer.setMiniBatchFraction(0.01);
        LDAModel ldaModel = new LDA().setK(80).setOptimizer(optimizer).run(corpus);
        // Output topics. Each is a distribution over words (matching word count vectors)
        System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize()
                + " words):");
        Matrix topics = ldaModel.topicsMatrix().transpose();
        PrintWriter pw = null;
        // ldaModel.
//        System.out.println(topics.toString());
        //  ldaModel.
// Load and parse the data
        // String testFilePath = "data/pubmed/2000now/pubmed_content_test.txt";

        JavaRDD<LabeledPoint> testData = MLUtils.loadLibSVMFile(sc.sc(), testFilePath).toJavaRDD();
        JavaPairRDD<Long, Vector> testCorpus = testData.mapToPair(
                new PairFunction<LabeledPoint, Long, Vector>() {

                    public Tuple2<Long, Vector> call(LabeledPoint point) {
                        return new Tuple2((long) point.label(), point.features());
                    }
                }
        );
        // System.out.println(Math.exp(((LocalLDAModel)ldaModel).logPerplexity(testCorpus)));
        try {
            pw = new PrintWriter(new FileWriter(outFolderPath + File.separator + "topics.txt"));

            for (int ii = 0; ii < topics.numRows(); ii++) {
                StringBuffer s = new StringBuffer();
                for (int jj = 0; jj < topics.numCols(); jj++) {
                    s.append(Double.toString(topics.apply(ii, jj)) + " ");
                }

                pw.write(s.toString() + "\n");
            }
            pw.close();
            pw = new PrintWriter(new FileWriter(outFolderPath + File.separator + "perp.txt"));
            double perp = Math.exp(((LocalLDAModel) ldaModel).logPerplexity(testCorpus));
            pw.write(perp + "\n");
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (pw != null)
                pw.close();
        }
// ldaModel.lo

        // ldaModel.save(sc.sc(), "myLDAModel");
        //  DistributedLDAModel sameModel = DistributedLDAModel.load(sc.sc(), "myLDAModel");
    }
}
