/**
 * Created by hvhuynh on 30/10/2015.
 */

import breeze.linalg.*;
import no.uib.cipr.matrix.sparse.*;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.DoubleArray;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.Intercept;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.SparseMatrix;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.bnpstat.maths.*;
import org.bnpstat.maths.SparseVector;
import org.bnpstat.mc2.MC2Data;
import org.bnpstat.mc2.MC2StochasticVariationalInference;
import org.bnpstat.mc2.MC2StochasticVariationalInferenceSpark;
import org.apache.spark.mllib.util.MLUtils;
import org.bnpstat.util.FastMatrixFunctions;
import org.bnpstat.util.Parallel;
import org.bnpstat.util.PerformanceTimer;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Int;
import scala.Tuple2;
import scala.runtime.BoxedUnit;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import no.uib.cipr.matrix.*;

public class ExternalLibTest {
    public static void main(String[] args) {
//        System.setProperty("hadoop.home.dir", "V:\\Code\\Java\\BNPStat\\hadoop-common-2.2.0-bin-master\\");

        // SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ").setMaster("local");
        // JavaSparkContext jsc = new JavaSparkContext(conf);
//        SparkContext jsc = new SparkContext(conf);

//        ArrayRealVector gt = new ArrayRealVector(3, 0);
//        gt.setEntry(0, 2);
//        gt.setEntry(1,4);
//        gt.setEntry(2,4);
//
//        System.out.println(gt.unitVector());
//        System.out.println(gt.getNorm());
//        System.out.println(gt.getL1Norm());
//        System.out.println(gt.mapDivideToSelf(gt.getL1Norm()));
        //MC2StochasticVariationalInferenceSpark.smallDataExp();
        //MC2StochasticVariationalInference.NIPSExperiment_AuthorContext();
        //MC2StochasticVariationalInferenceSpark.NIPSExperiment_AuthorContext();
//
//        String strContent = "F:\\MyPhD\\Code\\CSharp\\MC2_VU_CODE\\csdp shared\\csdp shared NIPS dataset\\content_nips.txt";
//        String strContext = "F:\\MyPhD\\Code\\CSharp\\MC2_VU_CODE\\csdp shared\\csdp shared NIPS dataset\\context_nips_author.txt";
//        String strOutMat = "D:\\Nips_Author_SVI_CSharp_SPARK\\";

        // MC2StochasticVariationalInferenceSpark.NIPSExperiment_AuthorContext_DataFromFile(strContent,strContext,strOutMat);

        /*
        JavaRDD<LabeledPoint> content =MLUtils.loadLibSVMFile(jsc.sc(), "D:/libsvm.dat").toJavaRDD();
        JavaRDD<LabeledPoint> context =MLUtils.loadLibSVMFile(jsc.sc(), "D:/libsvm1.txt").toJavaRDD();
        JavaRDD<MC2Data> data=content.map((point)->{
            ArrayList<Object> ss= new ArrayList();
            double[] values= point.features().toSparse().values();
            int[] indices= point.features().toSparse().indices();
            for (int i=0;i<indices.length;i++)
                for (int c=0;c<(int)values[i];c++)
                    ss.add(indices[i]);
            return new MC2Data((int)point.label(),null,ss);
        });
        data=data.union(context.map((point)->{
            return new MC2Data((int)point.label(),new SparseVector(point.features().toArray()),null);
        }));
        //org.apache.spark.mllib.linalg.SparseVector
       // examples.foreach((LabeledPoint point)-> {System.out.println(point);});
        JavaPairRDD<Integer, MC2Data> pairs= data.mapToPair((point)-> {return new Tuple2((int)point.id,point);});
        pairs=pairs.reduceByKey((MC2Data data1,MC2Data data2)-> {
            return data1.ss==null?new MC2Data(data1.id,data1.xx,data2.ss):new MC2Data(data1.id,data2.xx,data1.ss);
        });

        JavaRDD<MC2Data>  corpus=pairs.values();
        int[] numdata=new int[(int)corpus.count()];
        int c=0;
        corpus.collect().forEach(point-> {
            numdata[point.id]=point.ss.size();
        }
        );
*/
//        SparseVector v= new SparseVector(10);
//        v.addValue(1,10);
//        v.addValue(4,2);
//        v.addValue(9,4);
//
//
//        SparseVector v1= new SparseVector(10);
//        v1.addValue(2,1);
//        v1.addValue(4,4);
//        v1.addValue(5,2);
//        //v.mul(v1);
//        v.mul(4);
//
//        v.plus(v1);
// Collection of items to process in parallel
//        PerformanceTimer timer = new PerformanceTimer();
//
//        Collection<Integer> elems = new LinkedList<Integer>();
//       Collection<Double> elems1 = new LinkedList<Double>();
//
//        for (int i = 0; i < 100000; ++i) {
//            elems.add(i);
//           // elems1.add(i);
//        }
//        timer.start();
//        Parallel.For(elems,
//                // The operation to perform with each item
//                new Parallel.Operation<Integer>() {
//                    public void perform(Integer param) {
//                        elems1.add(Math.exp(param));
//                    };
//                });
//        timer.stop();
//
//        System.out.println(timer.getElaspedMiliSeconds());
//        timer.start();
//        Object[]temp= elems.toArray();
//        for (int i = 0; i < 100000; ++i) {
//            elems1.add(Math.exp((Integer)temp[i]));
//        }
//        timer.stop();
//
////        System.out.println(timer.getElaspedMiliSeconds());getElaspedMiliSeconds
//        int KK = 2;
//        int MM = 3;
//        int NN = 5;
//        int TT = 4;
//        ArrayList<DenseMatrix> data= new ArrayList<>();
//        for (int kk=0;kk<KK;kk++) {
//            DenseMatrix mat = DenseMatrix.zeros(TT,MM);
//            data.add(mat);
//        }
//        INDArray my3Arr= Nd4j.rand(new int[] {KK,TT,MM});
//        System.out.println(my3Arr);
//        System.out.println("**********************");
//
//        INDArray my2Arr= Nd4j.rand(new int[] {TT,MM});
//        System.out.println(my2Arr);
//        System.out.println("**********************");
//        INDArray my1Arr= Nd4j.rand(new int[] {MM});
//        System.out.println(my1Arr);
//        System.out.println("**********************");
//
//        INDArray myNewArr =my2Arr.broadcast(KK,TT,MM);
//        System.out.println(myNewArr);
//        System.out.println("**********************");
//
//
        int KK = 5;
        int MM = 4;
        int NN =10;
        int TT = 3;
        double[][][] qcc = new double[KK][TT][MM];
        double[][] EE =DoubleMatrix.rand(NN,MM).toArray2(); //new double[NN][MM];
        for(int kk=0;kk<KK;kk++)
            qcc[kk]=DoubleMatrix.rand(TT,MM).toArray2();

        PerformanceTimer innerTimer = new PerformanceTimer();
        innerTimer.start();
        double[][][] out1= multNDArray(qcc, EE);
        innerTimer.stop();
        System.out.println(innerTimer.getElaspedMiliSeconds());


//        innerTimer.start();
//        DoubleMatrix[] qccMat=new DoubleMatrix[qcc.length];
//        for (int ii = 0; ii < qcc.length; ii++) {
//            qccMat[ii]=new DoubleMatrix(qcc[ii]);
//        }
//        multNDArrayJBLAS(qccMat, EE);
//        innerTimer.stop();
//        System.out.println(innerTimer.getElaspedMiliSeconds());


        innerTimer.start();

        DoubleMatrix stick= new DoubleMatrix(KK,TT);
        stick.addi(0.1);
        double[][][] out2=multNDArrayJBLAS2(qcc, EE,stick.toArray2());

        innerTimer.stop();
        System.out.println(innerTimer.getElaspedMiliSeconds());
        System.out.println(innerTimer.getElaspedMiliSeconds());

//        DoubleMatrix mat= new DoubleMatrix(9,2);
//        DoubleMatrix matVal= DoubleMatrix.rand(3,2);
//        IntervalRange colRange=new IntervalRange(0,2);
//        double[][][] newMat=new double[3][3][2];
//        for(int ii=0;ii<3;ii++){
//            IntervalRange rowRange=new IntervalRange(3*ii,3*(ii+1));
//            mat.put(rowRange,colRange,matVal);
//        }
//        System.out.println(matVal);
//        System.out.println(mat);
//        for(int ii=0;ii<3;ii++){
//            IntervalRange rowRange=new IntervalRange(3*ii,3*(ii+1));
//            newMat[ii]=mat.get(rowRange,colRange).toArray2();
//        }
//
//       // System.out.println(mat);
//        double[][] newMat1=DoubleMatrix.ones(10,2).toArray2();
//        double[][][]newMat2= multNDArrayJBLAS2(newMat,newMat1);


        double[][][] qtt = new double[NN][KK][TT];
        double[] qzz =DoubleMatrix.rand(KK).toArray(); //new double[NN][MM];
        for(int nn=0;nn<NN;nn++)
            qtt[nn]=DoubleMatrix.rand(KK,TT).toArray2();

        double[][] out3=multNDArray(qtt, qcc,qzz);
        double[][] out4 = FastMatrixFunctions.multNDArrayJBLAS(qtt,qcc,qzz);
System.out.println("dada");

    }
    public static double[][] multNDArray(double[][][] mat1, double[][][] mat2,double[]mat3) {
        int NN=mat1.length;
        int KK= mat2.length;
        int TT=mat2[0].length;
        int MM=mat2[0][0].length;
        double[][] result= new double[NN][MM];

        for (int kk = 0; kk < KK; kk++) {
            // Computing the last TT
            for (int tt = 0; tt < TT; tt++) {

                for (int ii = 0; ii < NN; ii++) {

                    for (int mm = 0; mm < MM; mm++) {
                        result[ii][mm] += mat1[ii][kk][tt] * mat2[kk][tt][mm] * mat3[kk];
                    }

                }
            }
        }
        return result;
    }
    public static double[][][] multNDArray(double[][][] mat1, double[][] mat2) {
        int NN = mat2.length;
        int KK = mat1.length;
        int TT = mat1[0].length;
        int MM = mat1[0][0].length;
        double[][][] result = new double[NN][KK][TT];
        for (int ii = 0; ii < NN; ii++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int kk = 0; kk < KK; kk++) {
                    result[ii][kk][tt] = 0;
                    for (int mm = 0; mm < MM; mm++) {
                        result[ii][kk][tt] += mat1[kk][tt][mm] * mat2[ii][mm];
                    }
                    result[ii][kk][tt] += 0.1;
                }
            }
        }
        return result;
    }
    public static DoubleMatrix[] multNDArrayJBLAS(DoubleMatrix[] mat1, double[][] mat2) {
        int NN = mat2.length;
        int KK = mat1.length;
        DoubleMatrix[] result = new DoubleMatrix[NN];
        for (int kk = 0; kk < KK; kk++) {
            for (int ii = 0; ii < NN; ii++)
            result[ii] = mat1[kk].muliRowVector(new DoubleMatrix(mat2[ii])).columnSums();
        }
        return result;
    }
    public static DoubleMatrix multNDArrayJBLAS1(double[][] mat1, double[][] mat2) {

        DoubleMatrix result = new DoubleMatrix(mat1);
        DoubleMatrix temp = new DoubleMatrix(mat2);
        result.mmul(temp);
        return result;
    }
    public static double[][][] multNDArrayJBLAS2(double[][][] mat1, double[][] mat2) {

        int KK=mat1.length;
        int TT=mat1[0].length;
        int MM=mat1[0][0].length;
        int NN= mat2.length;
        DoubleMatrix temp1 = new DoubleMatrix(KK*TT,MM);
        IntervalRange colRange=new IntervalRange(0,MM);
        for(int kk=0;kk<KK;kk++){
            IntervalRange rowRange=new IntervalRange(TT*kk,TT*(kk+1));
            temp1.put(rowRange,colRange, new DoubleMatrix(mat1[kk]));
        }
        DoubleMatrix temp = new DoubleMatrix(mat2);

        DoubleMatrix temp2=temp.mmul(temp1.transpose());
        double[][][] result=new double[NN][KK][TT];
        colRange=new IntervalRange(0,KK*TT);
        for(int nn=0;nn<NN;nn++){
            IntervalRange rowRange=new IntervalRange(nn,nn+1);
            result[nn]=temp2.get(rowRange,colRange).reshape(TT,KK).transpose().toArray2();
        }
        return result;
    }

    public static double[][][] multNDArrayJBLAS2(double[][][] mat1, double[][] mat2,double[][]mat3) {

        int KK=mat1.length;
        int TT=mat1[0].length;
        int MM=mat1[0][0].length;
        int NN= mat2.length;
        DoubleMatrix temp1 = new DoubleMatrix(KK*TT,MM);
        IntervalRange colRange=new IntervalRange(0,MM);
        for(int kk=0;kk<KK;kk++){
            IntervalRange rowRange=new IntervalRange(TT*kk,TT*(kk+1));
            temp1.put(rowRange,colRange, new DoubleMatrix(mat1[kk]));
        }
        DoubleMatrix temp = new DoubleMatrix(mat2);

        DoubleMatrix temp2=temp.mmul(temp1.transpose());
        DoubleMatrix temp3=new DoubleMatrix(mat3);
        double[][][] result=new double[NN][KK][TT];
        colRange=new IntervalRange(0,KK*TT);
        for(int nn=0;nn<NN;nn++){
            IntervalRange rowRange=new IntervalRange(nn,nn+1);
            result[nn]=temp2.get(rowRange,colRange).reshape(TT,KK).transpose().addi(temp3).toArray2();
        }
        return result;
    }
}
