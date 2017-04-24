import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.bnpstat.mc2.MC2Data;
import org.bnpstat.stats.conjugacy.DoubleMultivariateData;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hvhuynh on 2/17/2017.
 */
public class ReadFormFileTest {
    public static void main(String[]args){
        String userDir=System.getProperty("user.dir");
        System.setProperty("hadoop.home.dir", userDir+"\\hadoop-common-2.2.0-bin-master\\");
        String contFilePath = "R:\\Collaborations-Deakin\\Nhat_Ho_Nested_K_means\\Nested_Kmeans_Code\\Python_code_Viet\\data\\LabelMeGT4patches.dat";
        String cxtFilePath="R:\\MyResearch\\Projects\\nestedKmeans\\code\\python\\data\\LabelMeGT4patches_context.dat";
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ").setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        JavaRDD<MC2Data> data=jsc.textFile(contFilePath).zipWithIndex().map(
                new Function<Tuple2<String, Long>, MC2Data>() {
                    @Override
                    public MC2Data call(Tuple2<String, Long> doc) throws Exception {
                        ArrayList<Object> ss = new ArrayList();
                        String[] points=doc._1.split(";");
                        for (int i = 0; i < points.length; i++){
                            String[] pointStrings=points[i].split(" ");
                            double[] temp= new double[pointStrings.length];
                            for (int j=0;j<pointStrings.length;j++)
                                temp[j]=Double.valueOf(pointStrings[j]);
                            ss.add(new DoubleMultivariateData(temp));
                        }

                        return new MC2Data(doc._2.intValue(),null,ss);
                    }
                }
        );
        JavaRDD<MC2Data> contextData=jsc.textFile(cxtFilePath).zipWithIndex().map(
                new Function<Tuple2<String, Long>, MC2Data>() {
                    @Override
                    public MC2Data call(Tuple2<String, Long> doc) throws Exception {
                        // Process missing data -- return null for missing
                        if (doc._1.isEmpty())
                            return new MC2Data(doc._2.intValue(),null,null);

                        String[] pointStrings=doc._1.split(";");
                        double[] temp= new double[pointStrings.length];
                        for (int i = 0; i < pointStrings.length; i++){
                                temp[i]=Double.valueOf(pointStrings[i]);
                        }
                        return new MC2Data(doc._2.intValue(),new DoubleMultivariateData(temp),null);
                    }
                }
        );
        int ngroups=(int)data.count();
        int[] numdata = new int[ngroups];
        data=data.union(contextData);
        // merging content with context record
        JavaPairRDD<Integer, MC2Data> pairs = data.mapToPair(
                new PairFunction<MC2Data, Integer, MC2Data>() {
                    @Override
                    public Tuple2<Integer, MC2Data> call(MC2Data point) throws Exception {
                        return new Tuple2((int) point.id, point);
                    }
                }
        );

        pairs = pairs.reduceByKey(
                new Function2<MC2Data, MC2Data, MC2Data>() {
                    @Override
                    public MC2Data call(MC2Data data1, MC2Data data2) throws Exception {
                        return data1.ss == null ? new MC2Data(data1.id, data1.xx, data2.ss) : new MC2Data(data1.id, data2.xx, data1.ss);
                    }
                }
        );
        JavaRDD<MC2Data> corpus = pairs.values();
        System.out.print("Done");

    }
}
