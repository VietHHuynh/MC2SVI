package org.bnpstat.mc2;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.bnpstat.maths.SparseVector;
import org.bnpstat.stats.conjugacy.*;
import org.bnpstat.util.FastMatrixFunctions;
import org.bnpstat.util.MatlabJavaConverter;
import org.bnpstat.util.PerformanceTimer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.codehaus.jackson.node.DoubleNode;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import scala.Tuple2;

import java.io.*;
import java.util.*;

/**
 * Created by hvhuynh on 11/9/2015.
 */

public class MC2StochasticVariationalInferenceSpark implements Serializable {

    // Predefined constant for distribution that data sampled from
    public final static int CATEGORICAL = 0;
    public final static int MULTINOMIAL = 1;
    public final static int UNIVARIATE_GAUSSIAN = 2;
    public final static int MULTIVARIATE_GAUSSIAN = 3;

    private static BayesianComponent q0;
    private static double aa;

    private static BayesianComponent p0;
    private static double ee;
    private static double gg;


    private static int ngroups;
    private static int[] numdata;
    private static JavaRDD<MC2Data> corpus;
    private static JavaRDD<MC2Data> testCorpus;

    // private ArrayList<Object> xx; //content data
    // private ArrayList<Object>[] ss; //context data
    private static int KK;
    private static int TT;
    private static int MM;

    // Model variables
    private static ArrayList<BayesianComponent> qq;
    private static ArrayList<BayesianComponent> pp;
    private static double[] qzz;
    private static double[][][] qcc;
    private static double[][] rhos;
    private static double[][][] zetas;
    private static double[][] varphis;
    private static boolean isZeroStart;

    // SparkContext
    JavaSparkContext sc;
    /**
     * Constructor
     */

    // List of static functions
    private static Function fMapLabeledPointToMC2CategoryDataContent = new Function<LabeledPoint, MC2Data>() {
        @Override
        public MC2Data call(LabeledPoint point) throws Exception {
//            int indShift= isZeroStart?1:0;
            ArrayList<Object> ss = new ArrayList();
            double[] values = point.features().toSparse().values();
            int[] indices = point.features().toSparse().indices();
            for (int i = 0; i < indices.length; i++)
                for (int c = 0; c < (int) values[i]; c++)
                    ss.add(indices[i]);
            return new MC2Data((int) point.label(), null, ss);
        }
    };

    private static Function fMapLabeledPointToMC2DataMultinomialContext = new Function<LabeledPoint, MC2Data>() {
        @Override
        public MC2Data call(LabeledPoint point) throws Exception {
//            int indShift= isZeroStart?1:0;
            double[] values = point.features().toSparse().values();
            int[] indices = point.features().toSparse().indices();
            int dim = ((MultinomialDirichlet) q0).getDim();
//            for(int ii=0;ii<indices.length;ii++)
//                indices[ii]-=indShift;

            SparseVector vec = new SparseVector(indices, values, dim);

            return new MC2Data((int) point.label(), vec, null);
        }
    };
    private static Function fMapLabeledPointToMC2DataGaussianContext = new Function<LabeledPoint, MC2Data>() {
        @Override
        public MC2Data call(LabeledPoint point) throws Exception {
//            int indShift= isZeroStart?1:0;
            double[] values = point.features().toArray();
            Double val = new Double(values[0]);
            return new MC2Data((int) point.label(), val, null);
        }
    };

    /**
     * For loading Multivariate Gaussian data
     */
    private static Function fMapTextToMC2MultivariateGaussianContent = new Function<Tuple2<String, Long>, MC2Data>() {
        @Override
        public MC2Data call(Tuple2<String, Long> doc) throws Exception {
            ArrayList<Object> ss = new ArrayList();
            String[] points = doc._1.split(";");
            for (int i = 0; i < points.length; i++) {
                String[] pointStrings = points[i].split(" ");
                double[] temp = new double[pointStrings.length];
                for (int j = 0; j < pointStrings.length; j++)
                    temp[j] = Double.valueOf(pointStrings[j]);
                ss.add(new DoubleMultivariateData(temp));
            }

            return new MC2Data(doc._2.intValue(), null, ss);
        }
    };

    private static Function fMapTextToMC2MultivariateGaussianContext = new Function<Tuple2<String, Long>, MC2Data>() {
        @Override
        public MC2Data call(Tuple2<String, Long> doc) throws Exception {
            // Process missing data -- return null for missing
            if (doc._1.isEmpty())
                return new MC2Data(doc._2.intValue(), null, null);

            String[] pointStrings = doc._1.split(";");
            double[] temp = new double[pointStrings.length];
            for (int i = 0; i < pointStrings.length; i++) {
                temp[i] = Double.valueOf(pointStrings[i]);
            }
            return new MC2Data(doc._2.intValue(), new DoubleMultivariateData(temp), null);
        }
    };

    private static PairFunction pfMapIDtoKey = new PairFunction<MC2Data, Integer, MC2Data>() {
        @Override
        public Tuple2<Integer, MC2Data> call(MC2Data point) throws Exception {
            return new Tuple2((int) point.id, point);
        }
    };
    public static Function2 f2CombineMC2Data = new Function2<MC2Data, MC2Data, MC2Data>() {
        @Override
        public MC2Data call(MC2Data data1, MC2Data data2) throws Exception {
            return data1.ss == null ? new MC2Data(data1.id, data1.xx, data2.ss) : new MC2Data(data1.id, data2.xx, data1.ss);
        }
    };

    public MC2StochasticVariationalInferenceSpark(int KKTrun, int TTTrun, int MMTrun, double epsison, double alpha, double beta, BayesianComponent q0Context, BayesianComponent p0Content, JavaSparkContext jsc) {
        KK = KKTrun;
        TT = TTTrun;
        MM = MMTrun;
        ee = epsison;
        aa = alpha;
        gg = beta;
        q0 = q0Context;
        p0 = p0Content;

        // Initialize components
        qq = new ArrayList<BayesianComponent>(); //context
        pp = new ArrayList<BayesianComponent>(); //content

        // Context
        for (int kk = 0; kk < KK; kk++) {
            qq.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            pp.add((BayesianComponent) p0.clone());
        }
        // SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ").setMaster("local");
        sc = jsc;
    }

    /**
     * Initialize parameters and hyperparameters for the VB learning
     */
    public void initialize() {
        Random rnd = new Random();
        rnd.setSeed(6789);

        ArrayList<Object> xx = new ArrayList<Object>(); //content data
        ArrayList<Object>[] ss = new ArrayList[ngroups]; //context data

        /*corpus.collect().forEach((MC2Data doc) -> {
                xx.add(doc.xx);
               ss[doc.id]=doc.ss;
        });*/

        //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
        for (MC2Data doc : corpus.collect()) {
            xx.add(doc.xx);
            ss[doc.id] = doc.ss;
        }

        //Intialize context data
        for (int j = 0; j < ngroups; j++) {
            ArrayRealVector gt = new ArrayRealVector(KK, 0);
            int[] singletons = new int[KK];
            try {
                for (int k = 0; k < KK; k++) {
                    gt.setEntry(k, qq.get(k).logPredictive(xx.get(j)));
                    singletons[k] = k;
                }

                double gtMax = gt.getMaxValue();
                for (int k = 0; k < KK; k++) {
                    gt.setEntry(k, FastMath.exp(gt.getEntry(k) - gtMax));
                }

                gt.mapDivideToSelf(gt.getL1Norm());
                int kk = (new EnumeratedIntegerDistribution(singletons, gt.toArray())).sample();
                MultinomialDirichlet temp = (MultinomialDirichlet) qq.get(kk);
                temp.add(xx.get(j));
                //qq.get(kk).add(xx.get(j));
            } catch (Exception e) {
                e.printStackTrace();
            }
            //Intialize content data

            for (int i = 0; i < numdata[j]; i++) {
                try {
                    gt = new ArrayRealVector(MM, 0);
                    singletons = new int[MM];
                    for (int m = 0; m < MM; m++) {
                        gt.setEntry(m, pp.get(m).logPredictive(ss[j].get(i)));
                        singletons[m] = m;
                    }
                    double gtMax = gt.getMaxValue();
                    for (int m = 0; m < MM; m++) {
                        gt.setEntry(m, FastMath.exp(gt.getEntry(m) - gtMax));
                    }
                    gt.mapDivideToSelf(gt.getL1Norm());
                    int mm = (new EnumeratedIntegerDistribution(singletons, gt.toArray())).sample();

                    pp.get(mm).add(ss[j].get(i));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        // Initialize stick-breaking
        // rhos
        rhos = new double[KK - 1][2];
        double sumrhos = rnd.nextInt(1); // last KK
        for (int kk = KK - 2; kk >= 0; kk--) {
            rhos[kk][0] = 1 + rnd.nextInt(1);
            rhos[kk][1] = gg + sumrhos;
            sumrhos += rhos[kk][0] - 1;
        }

        //zetas
        zetas = new double[KK][TT - 1][2];
        for (int kk = 0; kk < KK; kk++) {
            double sumzetas = rnd.nextInt(1); // last TT
            for (int tt = TT - 2; tt >= 0; tt--) {
                zetas[kk][tt][0] = 1 + rnd.nextInt(1);
                zetas[kk][tt][1] = aa + sumzetas;
                sumzetas += zetas[kk][tt][0] - 1;
            }
        }

        //varphis
        varphis = new double[MM - 1][2];
        double sumvarphis = rnd.nextInt(1); // last KK
        for (int mm = MM - 2; mm >= 0; mm--) {
            varphis[mm][0] = 1 + rnd.nextInt(1);
            varphis[mm][1] = ee + sumvarphis;
            sumvarphis += varphis[mm][0] - 1;
        }

        // qcc
        // qcc
        double eps = 0.000001;
        qcc = new double[KK][TT][MM];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM; mm++)
                    qcc[kk][tt][mm] = eps / (eps * (MM - 1) + 1);
                qcc[kk][tt][rnd.nextInt(MM)] = 1 / (eps * (MM - 1) + 1);
            }
        }

    }

    public void initializeRandom() {
        Random rnd = new Random();
//        rnd.setSeed(6789);

        //Initialize context data by random
        for (int kk = 0; kk < KK; kk++) {
            //int[] randData= new int[((MultinomialDirichlet)qq.get(0)).getDim()];
            SparseVector randData = new SparseVector(((MultinomialDirichlet) qq.get(0)).getDim());
            for (int ii = 0; ii < randData.getLength(); ii++)
                randData.setValue(ii, rnd.nextDouble());
            try {
                qq.get(kk).add(randData);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
        for (int mm = 0; mm < MM; mm++) {
//            int[] randData= new int[((MultinomialDirichlet)pp.get(0)).getDim()];
            SparseVector randData = new SparseVector(((MultinomialDirichlet) pp.get(0)).getDim());

            for (int ii = 0; ii < randData.getLength(); ii++)
                randData.setValue(ii, rnd.nextDouble());
            try {
                pp.get(mm).add(randData);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
        // Initialize stick-breaking
        // rhos
        rhos = new double[KK - 1][2];
        double sumrhos = rnd.nextDouble(); // last KK
        for (int kk = KK - 2; kk >= 0; kk--) {
            rhos[kk][0] = 1 + rnd.nextDouble();
            rhos[kk][1] = gg + sumrhos;
            sumrhos += rhos[kk][0] - 1;
        }

        //zetas
        zetas = new double[KK][TT - 1][2];
        for (int kk = 0; kk < KK; kk++) {
            double sumzetas = rnd.nextDouble(); // last TT
            for (int tt = TT - 2; tt >= 0; tt--) {
                zetas[kk][tt][0] = 1 + rnd.nextDouble();
                zetas[kk][tt][1] = aa + sumzetas;
                sumzetas += zetas[kk][tt][0] - 1;
            }
        }

        //varphis
        varphis = new double[MM - 1][2];
        double sumvarphis = rnd.nextDouble(); // last KK
        for (int mm = MM - 2; mm >= 0; mm--) {
            varphis[mm][0] = 1 + rnd.nextDouble();
            varphis[mm][1] = ee + sumvarphis;
            sumvarphis += varphis[mm][0] - 1;
        }

        qcc = new double[KK][TT][MM];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                ArrayRealVector gt = new ArrayRealVector(MM, 0);
                for (int mm = 0; mm < MM; mm++) {
                    gt.setEntry(mm, rnd.nextDouble());
                }
                qcc[kk][tt] = gt.mapDivideToSelf(gt.getL1Norm()).toArray();
            }
        }
    }

    public void initializeMultivariateGaussianData() {
        Random rnd = new Random();
//        rnd.setSeed(6789);

        //Initialize context data by random
        for (int kk = 0; kk < KK; kk++) {
            double[] vals = new double[((MultivariateGaussianGamma) qq.get(0)).getDim()];
            for (int ii = 0; ii < vals.length; ii++)
                vals[ii] = rnd.nextDouble();
            DoubleMultivariateData randData = new DoubleMultivariateData(vals);
            try {
                qq.get(kk).add(randData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        //Initialize content data by random
        for (int mm = 0; mm < MM; mm++) {

            double[] vals = new double[((MultivariateGaussianGamma) pp.get(0)).getDim()];
            for (int ii = 0; ii < vals.length; ii++)
                vals[ii] = 1.5 - 3 * rnd.nextDouble();//(0.5-rnd.nextDouble())/5;
            DoubleMultivariateData randData = new DoubleMultivariateData(vals);
            try {
                pp.get(mm).add(randData);
            } catch (Exception e) {
                e.printStackTrace();
            }
            double s0 = ((MultivariateGaussianGamma) p0).getS0();
            double a0 = ((MultivariateGaussianGamma) p0).getA0();
            double b0 = ((MultivariateGaussianGamma) p0).getB0();
            double[] mu00 = {1.40908128150013, 0.854327142266423, 0.0489343413177432, -0.0664661559750310, 0.233447663821825, -0.00813120589803062, 0.0126416771674140, -0.0217459034125060, 0.0300444078499502, -0.0563024103284583, 0.00245741358075010, 0.00446191691182997, -0.0572709997885438, 0.0946495666242964, 0.113749858700504, -0.0181551045973821, -0.0162812071716361, -0.0474763668208366, 0.0333542281781222, -0.00812311756359505, -0.0116020846842136, 0.00799018399359236, 0.00427304299331167, 0.000483058811385480, -0.0100593081669415, 0.0361111027851057, -0.00682644239313987, -0.0106909763246796, 0.00999009791785319, -0.00825521965228203};
            pp.set(0, new MultivariateGaussianGamma(mu00, s0, a0, b0));
            double[] mu01 = {0.497586636534916, 0.722155373109461, 0.319073464201126, 0.245093209547183, -0.0671870819004828, -0.0171813833503312, 0.0348801349033742, -0.0379906504563266, 0.0524298937111367, -0.0871201913587199, 0.00566069339989366, 0.00194849934009628, -0.0927945944527686, 0.158334149167843, 0.0800045916284517, -0.0119121859576133, -0.00687077376617092, -0.0475826320700717, 0.0210729061092173, -0.00861462338918690, -0.00950251748139646, 0.0100136798687219, -0.0192799689680118, -0.00988888122844712, 0.00737172525319536, 0.0352573996055497, -0.0190220691674320, 9.40555284289816e-05, -0.00467806632443222, -0.000410748392687959};
            pp.set(1, new MultivariateGaussianGamma(mu01, s0, a0, b0));
            double[] mu02 = {0.923933142068854, 1.19286731487221, 0.324928774123397, 0.0151889980788116, 0.155899212462104, 0.139259667976370, 0.0434271669807312, -0.0416006446496238, 0.0310591358065850, -0.0298175235402637, 0.00682996574670982, -0.0213099819067659, -0.110769635392554, 0.105720889851828, 0.122221985110091, -0.00895915700646040, -0.00735949177158252, -0.0397944928973731, 0.00401210794857474, -0.00699356910320620, -0.00293505756499193, 0.00833486377891127, -0.0404363404998627, -0.0144397771194378, 0.0227966925337190, 0.00953666861664902, -0.0102399622561709, -0.00907326243194570, 0.00981819355426097, -0.0161504617096919};
            pp.set(2, new MultivariateGaussianGamma(mu02, s0, a0, b0));
            double[] mu03 = {1.25998960818941, 0.440279242062598, 0.165597307659302, -0.154317679080307, 0.0139515063303280, -0.00769865927458366, 0.0348141565564770, -0.0263741545867165, 0.0573119391913299, -0.0493305423579399, 0.00292660182667118, -0.0188847972262769, -0.124372624780617, 0.121720550774010, 0.107478574898677, -0.00612812186868375, -0.00706299432815830, -0.0299959501485201, 0.00442315706020140, -0.0128055112670958, -0.00822218507900356, 0.00716131240997626, -0.0203835200041425, -0.00843106563238946, 0.0164947070828838, 0.0124320682814910, -0.0149475084100386, -0.0137889289206413, 0.00582154546442756, -0.00544848796836290};
            pp.set(3, new MultivariateGaussianGamma(mu03, s0, a0, b0));
            double[] mu04 = {0.724093051556879, 0.740558356605706, 0.485449235988723, -0.258502747136107, 0.159228999291396, 0.0337260236971199, 0.0558768628422227, -0.0554928805673269, 0.0628753891579380, -0.0587952143385931, 0.00579124426148192, -0.0152424290096552, -0.110376555942581, 0.111506277837893, 0.0966750695329618, -0.0149350965401492, -0.000650091199144117, -0.0429891172505032, 0.0172769307373573, 0.00424154510571676, -0.00616723882899092, 0.0127897819822928, -0.0168837394629094, -0.00870610681427844, 0.00895363000995318, 0.0261470855634830, -0.0142734619193198, -0.00896199203764444, 0.00481754307855203, -0.00695967838424737};
            pp.set(4, new MultivariateGaussianGamma(mu04, s0, a0, b0));
            double[] mu05 = {0.954104113548396, 0.801722609706068, 0.873059638784410, -0.0821000189073379, 0.155835799651055, -0.114881528423832, -0.00377193911613096, -0.0189586605947943, 0.0211218739367205, -0.0404225849205695, 0.000443035281118988, -0.0111682270882039, -0.118028355100659, 0.113609533503355, 0.115702181564029, -0.00799115525759648, -0.00875007938403994, -0.0322914425649968, 0.0154657815062363, -0.00569420269542750, -0.00572818801627857, 0.00396394858005581, 0.00159121156029903, -0.00150318500361370, -0.000462963719803010, 0.0191012027405015, -0.0179399154347939, -0.0125897281812094, 0.00408775332411985, 0.000115587066822539};
            pp.set(5, new MultivariateGaussianGamma(mu05, s0, a0, b0));
            double[] mu06 = {1.22507496922964, 0.883348014341557, 0.486404313906566, -0.174520686828622, 0.0248608332971746, -0.0142367828041372, 0.0554814244775878, -0.0579479732776960, 0.0774760194763448, -0.106158969701471, 0.0175699581219417, -0.00993236682196473, -0.0791702184937562, 0.106702591547300, 0.106564187729363, -0.0124561976106629, -0.00785648250263302, -0.0368261857926073, 0.0132376543878921, -0.0119246619858499, -0.00726620067386018, 0.00202520270878101, -0.0212869756946872, -0.00955691361632574, 0.0120879107399396, 0.0235388551664599, -0.0235118907427668, -0.00836980435913147, 0.00686346991238304, -0.00742089767573644};
            pp.set(6, new MultivariateGaussianGamma(mu06, s0, a0, b0));
            double[] mu07 = {0.291046279751683, 0.413243053380153, 0.274595904095386, -0.201638052021892, 0.177533207939436, 0.0441233347329184, 0.0399830904403519, -0.0430971399856573, 0.0598507951641730, -0.0792152242889368, 0.00969862526161188, -0.00252783217972202, -0.0569942085766389, 0.0641907956894040, 0.144514946976457, -0.0135741197221929, -0.0201075638104964, -0.0387125474082492, 0.0162832343690358, -0.0174623687603955, -0.00585819266142034, 0.00100527434383486, -0.0190630725794072, -0.00718509732631535, 0.0122367756165116, 0.0275845788631418, -0.0162304549427763, -0.0182446292689029, 0.0159032257525924, -0.0142319772237056};
            pp.set(7, new MultivariateGaussianGamma(mu07, s0, a0, b0));
            double[] mu08 = {1.21836739602966, 0.411610284276771, 0.490925290662347, 0.0975679476595851, 0.328732616758070, 0.0169399504199074, 0.0840414226731199, -0.0558436274962436, 0.114202508240676, -0.0770038190313175, 0.00194933438139654, -0.0252596153298798, -0.109460805310920, 0.0831208479566188, 0.0949587727626392, -0.0192041818026230, 0.0188861974310988, -0.0560534149047173, 0.0219813435783364, 0.0244443462498331, -0.000318776200004446, 0.00976899510235616, -0.0317328025005207, -0.0105902086528133, 0.0218959864250113, 0.0169236280414500, -0.0230987755927148, -0.00752989939422339, 0.000820358554766478, 0.00719876025085689};
            pp.set(8, new MultivariateGaussianGamma(mu08, s0, a0, b0));
            double[] mu09 = {1.36627461529707, 0.308854881517554, 0.795144219972646, -0.0212548580993457, 0.0643486235849491, 0.185106650039569, 0.0279307903518807, -0.0621788208531054, -0.0225182327838530, -0.0463871433857195, 0.00937204333483623, -0.00274256888641151, -0.0783122638042341, 0.108088677576786, 0.115601808177015, -0.0107959550779983, -0.0134372791590746, -0.0448443413886456, 0.0129458896928378, -0.0215261262010428, -0.00181338937128963, 0.0116097331563167, -0.0342313227318418, -0.0123444042413280, 0.0190876450753210, 0.0332946689879622, -0.00758980441201448, -0.00587051065206950, 0.0108513727918273, -0.0302182601546420};
            pp.set(9, new MultivariateGaussianGamma(mu09, s0, a0, b0));

        }
        // Initialize stick-breaking
        // rhos
        rhos = new double[KK - 1][2];
        double sumrhos = rnd.nextDouble(); // last KK
        for (int kk = KK - 2; kk >= 0; kk--) {
            rhos[kk][0] = 1 + rnd.nextDouble();
            rhos[kk][1] = gg + sumrhos;
            sumrhos += rhos[kk][0] - 1;
        }

        //zetas
        zetas = new double[KK][TT - 1][2];
        for (int kk = 0; kk < KK; kk++) {
            double sumzetas = rnd.nextDouble(); // last TT
            for (int tt = TT - 2; tt >= 0; tt--) {
                zetas[kk][tt][0] = 1 + rnd.nextDouble();
                zetas[kk][tt][1] = aa + sumzetas;
                sumzetas += zetas[kk][tt][0] - 1;
            }
        }

        //varphis
        varphis = new double[MM - 1][2];
        double sumvarphis = rnd.nextDouble(); // last KK
        for (int mm = MM - 2; mm >= 0; mm--) {
            varphis[mm][0] = 1 + rnd.nextDouble();
            varphis[mm][1] = ee + sumvarphis;
            sumvarphis += varphis[mm][0] - 1;
        }

        qcc = new double[KK][TT][MM];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                ArrayRealVector gt = new ArrayRealVector(MM, 0);
                for (int mm = 0; mm < MM; mm++) {
                    gt.setEntry(mm, rnd.nextDouble());
                }
                qcc[kk][tt] = gt.mapDivideToSelf(gt.getL1Norm()).toArray();
            }
        }
    }

    /**
     * Load data into local variables xx, and ss
     *
     * @param inXX - data loaded into xx: a list of Object
     * @param inSS - data loaded into ss: a two-dimension list of double
     */
    public void loadData(Object[] inXX, double[][] inSS) {
        ngroups = inXX.length;
        numdata = new int[ngroups];
        ArrayList<MC2Data> corpusList = new ArrayList(ngroups);
        for (int jj = 0; jj < ngroups; jj++) {
            ArrayList<Object> tempSS = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                tempSS.add(inSS[jj][ii]);
            }
            corpusList.add(jj, new MC2Data(jj, inXX[jj], tempSS));

        }
        corpus = sc.parallelize(corpusList);
    }

    /**
     * Load data into local variables xx, and ss
     *
     * @param inXX - data loaded into xx: a list of int
     * @param inSS - data loaded into ss: a two-dimension list of double
     */
    public void loadData(int[] inXX, double[][] inSS) {
        ngroups = inXX.length;
        numdata = new int[ngroups];
        ArrayList<MC2Data> corpusList = new ArrayList(ngroups);
        for (int jj = 0; jj < ngroups; jj++) {
            ArrayList<Object> tempSS = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                tempSS.add(inSS[jj][ii]);
            }
            corpusList.add(jj, new MC2Data(jj, inXX[jj], tempSS));

        }
        corpus = sc.parallelize(corpusList);
    }

    /**
     * Load data into local variables xx, and ss
     *
     * @param inXX - data loaded into xx: a list of int
     * @param inSS - data loaded into ss: a two-dimension list of int
     */
    public void loadData(int[] inXX, int[][] inSS) {
        ngroups = inXX.length;
        numdata = new int[ngroups];
        ArrayList<MC2Data> corpusList = new ArrayList(ngroups);
        for (int jj = 0; jj < ngroups; jj++) {
            ArrayList<Object> tempSS = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                tempSS.add(inSS[jj][ii]);
            }
            corpusList.add(jj, new MC2Data(jj, inXX[jj], tempSS));

        }
        corpus = sc.parallelize(corpusList);
    }

    /**
     * Load data into local variables xx, and ss
     *
     * @param inXX - data loaded into xx: a list of Sparse Vector
     * @param inSS - data loaded into ss: a two-dimension list of int
     */
    public void loadData(SparseVector[] inXX, int[][] inSS) {
        ngroups = inXX.length;
        numdata = new int[ngroups];

        ngroups = inXX.length;
        numdata = new int[ngroups];
        ArrayList<MC2Data> corpusList = new ArrayList(ngroups);
        for (int jj = 0; jj < ngroups; jj++) {
            ArrayList<Object> tempSS = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                tempSS.add(inSS[jj][ii]);
            }
            corpusList.add(jj, new MC2Data(jj, inXX[jj], tempSS));

        }
        corpus = sc.parallelize(corpusList);
    }

    /**
     * Load large data set into corpus where data in libsvm format (sparse vector)
     *
     * @param contFilePath - content data file in libsvm format
     * @param cxtFilePath  - context data file in libsvm format
     */
    public void loadData(String contFilePath, String cxtFilePath, boolean isZeroStart, String cxtType) throws Exception {
        this.isZeroStart = isZeroStart;
        JavaRDD<LabeledPoint> content = MLUtils.loadLibSVMFile(sc.sc(), contFilePath).toJavaRDD();
        JavaRDD<LabeledPoint> context = MLUtils.loadLibSVMFile(sc.sc(), cxtFilePath).toJavaRDD();
        ngroups = (int) content.count();
        numdata = new int[ngroups];

        /*JavaRDD<MC2Data> data=content.map((point)->{
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
        corpus=pairs.values();

        corpus.collect().forEach(point-> {
            numdata[point.id]=point.ss.size();
        }
        );*/
        //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
        JavaRDD<MC2Data> data = content.map(fMapLabeledPointToMC2CategoryDataContent);
        if (cxtType.equals("Multinomial"))
            data = data.union(context.map(fMapLabeledPointToMC2DataMultinomialContext));
        else if (cxtType.equals("Gaussian"))
            data = data.union(context.map(fMapLabeledPointToMC2DataGaussianContext));
        else
            throw new Exception("The context type is not defined yet which is Multinomial or Gaussian. Check mc2.contextType in config file!");

        //org.apache.spark.mllib.linalg.SparseVector
        // examples.foreach((LabeledPoint point)-> {System.out.println(point);});
        JavaPairRDD<Integer, MC2Data> pairs = data.mapToPair(pfMapIDtoKey);

        pairs = pairs.reduceByKey(f2CombineMC2Data);
        corpus = pairs.values();
        List allData = corpus.collect();
        for (Object point : allData) {
            numdata[((MC2Data) point).id] = ((MC2Data) point).ss.size();
        }
        corpus.cache();

    }

    public void loadData(String contFilePath, String cxtFilePath, String testContFilePath, String testCxtFilePath, boolean isZeroStart) {
        this.isZeroStart = isZeroStart;
        JavaRDD<LabeledPoint> content = MLUtils.loadLibSVMFile(sc.sc(), contFilePath).toJavaRDD();
        JavaRDD<LabeledPoint> context = MLUtils.loadLibSVMFile(sc.sc(), cxtFilePath).toJavaRDD();
        ngroups = (int) content.count();
        numdata = new int[ngroups];

        /*JavaRDD<MC2Data> data=content.map((point)->{
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
        corpus=pairs.values();

        corpus.collect().forEach(point-> {
            numdata[point.id]=point.ss.size();
        }
        );*/
        //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
        JavaRDD<MC2Data> data = content.map(fMapLabeledPointToMC2CategoryDataContent);
        data = data.union(context.map(fMapLabeledPointToMC2DataMultinomialContext));

        //org.apache.spark.mllib.linalg.SparseVector
        // examples.foreach((LabeledPoint point)-> {System.out.println(point);});
        JavaPairRDD<Integer, MC2Data> pairs = data.mapToPair(pfMapIDtoKey);

        pairs = pairs.reduceByKey(f2CombineMC2Data);
        corpus = pairs.values();
        List allData = corpus.collect();
        for (Object point : allData) {
            numdata[((MC2Data) point).id] = ((MC2Data) point).ss.size();
        }
        corpus.cache();

        JavaRDD<LabeledPoint> testContent = MLUtils.loadLibSVMFile(sc.sc(), testContFilePath).toJavaRDD();
        JavaRDD<LabeledPoint> testContext = MLUtils.loadLibSVMFile(sc.sc(), testCxtFilePath).toJavaRDD();


        //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
        JavaRDD<MC2Data> testData = testContent.map(fMapLabeledPointToMC2CategoryDataContent);
        data = data.union(testContext.map(fMapLabeledPointToMC2DataMultinomialContext));

        //org.apache.spark.mllib.linalg.SparseVector
        // examples.foreach((LabeledPoint point)-> {System.out.println(point);});
        JavaPairRDD<Integer, MC2Data> teatPairs = data.mapToPair(pfMapIDtoKey);

        teatPairs = teatPairs.reduceByKey(f2CombineMC2Data);
        testCorpus = teatPairs.values();
//       allData =testCorpus.collect();
//        for(Object point:allData){
//            numdata[((MC2Data)point).id]=((MC2Data)point).ss.size();
//        }
        testCorpus.cache();

    }

    public void loadMultivariateGaussianData(String contFilePath, String cxtFilePath) throws Exception {

        JavaRDD<MC2Data> contentData = sc.textFile(contFilePath, 1).zipWithIndex().map(fMapTextToMC2MultivariateGaussianContent
//                new Function<Tuple2<String, Long>, MC2Data>() {
//                    @Override
//                    public MC2Data call(Tuple2<String, Long> doc) throws Exception {
//                        ArrayList<Object> ss = new ArrayList();
//                        String[] points=doc._1.split(";");
//                        for (int i = 0; i < points.length; i++){
//                            String[] pointStrings=points[i].split(" ");
//                            double[] temp= new double[pointStrings.length];
//                            for (int j=0;j<pointStrings.length;j++)
//                                temp[j]=Double.valueOf(pointStrings[j]);
//                            ss.add(new DoubleMultivariateData(temp));
//                        }
//
//                        return new MC2Data(doc._2.intValue(),null,ss);
//                    }
//                }
        );
        JavaRDD<MC2Data> contextData = sc.textFile(cxtFilePath).zipWithIndex().map(
                fMapTextToMC2MultivariateGaussianContext
        );
        ngroups = (int) contentData.count();
        numdata = new int[ngroups];
        contentData = contentData.union(contextData);
        // merging content with context record
        JavaPairRDD<Integer, MC2Data> pairs = contentData.mapToPair(pfMapIDtoKey
//                new PairFunction<MC2Data, Integer, MC2Data>() {
//                    @Override
//                    public Tuple2<Integer, MC2Data> call(MC2Data point) throws Exception {
//                        return new Tuple2((int) point.id, point);
//                    }
//                }
        );

        pairs = pairs.reduceByKey(
                f2CombineMC2Data
//                new Function2<MC2Data, MC2Data, MC2Data>() {
//                    @Override
//                    public MC2Data call(MC2Data data1, MC2Data data2) throws Exception {
//                        return data1.ss == null ? new MC2Data(data1.id, data1.xx, data2.ss) : new MC2Data(data1.id, data2.xx, data1.ss);
//                    }
//                }
        );
        corpus = pairs.values();

        List allData = corpus.collect();
        for (Object point : allData) {
            numdata[((MC2Data) point).id] = ((MC2Data) point).ss.size();
        }
        corpus.cache();
    }

    /**
     * Computing stick breaking expectation with DP truncation
     *
     * @param hyperprams - hyper parameters of beta distributions for sticks
     * @return
     */
    private static double[] computeStickBreakingExpectation(double[][] hyperprams) throws Exception {
        if (hyperprams[0].length != 2)
            throw new Exception("The dimension of hyperparameters is not correct");
        int KK = hyperprams.length + 1;
        double[] EStick = new double[KK];

        double temp = Gamma.digamma(hyperprams[0][0] + hyperprams[0][1]);
        EStick[0] = Gamma.digamma(hyperprams[0][0]) - temp;// The first stick

        double sum = Gamma.digamma(hyperprams[0][1]) - temp;
        for (int kk = 1; kk < KK - 1; kk++) {
            temp = Gamma.digamma(hyperprams[kk][0] + hyperprams[kk][1]);
            EStick[kk] = Gamma.digamma(hyperprams[kk][0]) - temp + sum;
            sum += Gamma.digamma(hyperprams[kk][1]) - temp;
        }
        EStick[KK - 1] = sum;// The last stick
        return EStick;
    }


    /**
     * SVI inference for document of which data are categorical
     *
     * @param numIter      - the number of iterations (epoches)
     * @param batchSize    - mini batch size
     * @param varrho       - part of learning rate
     * @param iota         - part of learning rate (power)
     * @param strOutFolder - path to output folder
     */
    public static void sviCategoryOptimizer(int numIter, int batchSize, double varrho, double iota, String strOutFolder) {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++) {
            batchCount[i] = batchSize;
        }
        if ((ngroups - ngroups / batchSize * batchSize) > 0)

            batchCount[nBatches - 1] = ngroups - ngroups / batchSize * batchSize;
        else
            batchCount[nBatches - 1] = batchSize;

        Random rnd = new Random();
        rnd.setSeed(6789);
        /**
         * Start iterations
         */
        for (int iter = 1; iter <= numIter; iter++) {
            System.out.println("Iteration " + iter + " K=" + KK + " M=" + MM + " elapse=" + elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++) {
                timer.start();
                System.out.println("\t Running mini-batch " + ba + " from document # " + (noProcessedDocs + 1) + " to " + (noProcessedDocs + batchCount[ba]) + ", elapse= " + elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick = null;
                double[][] Elowerstick = null;
                double[] Esharestick = null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick = new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                HashMap<Integer, double[]> cachedContEloglik = new HashMap<Integer, double[]>();

                PerformanceTimer myTimer = new PerformanceTimer();
                //double total = 0;
                myTimer.start();
                final int startDoc = noProcessedDocs; //inclusive
                final int endDoc = noProcessedDocs + batchCount[ba]; //exclusive

               /* JavaRDD docChunk=corpus.filter(doc -> (doc.id>=startDoc && doc.id<endDoc));

                //Construct the key - word ids - for cachedContEloglik
                docChunk.collect().forEach((Object doc) -> {
                    ((MC2Data)doc).ss.forEach((Object word) -> {
                                if (!cachedContEloglik.containsKey((Integer) word)) {
                                    cachedContEloglik.put((Integer) word, null);
                                }
                            }
                    );
                });*/

                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                JavaRDD docChunk = corpus.filter(
                        new Function<MC2Data, Boolean>() {
                            @Override
                            public Boolean call(MC2Data doc) throws Exception {
                                return (doc.id >= startDoc && doc.id < endDoc);
                            }
                        }
                );
                //Construct the key - word ids - for cachedContEloglik
                for (Object doc : docChunk.collect()) {
                    for (Object word : ((MC2Data) doc).ss) {
                        if (!cachedContEloglik.containsKey((Integer) word)) {
                            cachedContEloglik.put((Integer) word, null);
                        }
                    }
                }

                myTimer.stop();
                System.out.println("\tRunning time for  getting word ids: " + myTimer.getElaspedSeconds());
                myTimer.start();
                ArrayList<Integer> keys = new ArrayList<Integer>(cachedContEloglik.keySet());
                double[][] tempEll = MultinomialDirichlet.expectationLogLikelihood(keys, pp);
                for (int ii = 0; ii < keys.size(); ii++)
                    cachedContEloglik.put(keys.get(ii), tempEll[ii]);
                myTimer.stop();
                System.out.println("\tRunning time for  computing expectation: " + myTimer.getElaspedSeconds());





               /*  int tempCount=batchCount[ba];
                double[] tempEtopStick =EtopStick;
                double[][] tempElowerstick =Elowerstick;
                double[] tempEsharestick = Esharestick;*/
                //int tempKK=KK;
                // int tempTT=TT;
                //  int tempMM=MM;


/*                MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions((listDocs)->{
                   //System.out.println(((MC2Data) doc).id);
                     // test();
                   // return null;
                   // return computeParamEachDoc((MC2Data)doc,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   ArrayList output=new ArrayList();
                   output.add(temp);
                   return output;
                }).reduce((node1, node2)->{return reduceOutput((MC2MapReduceOutput) ((ArrayList)node1).get(0),(MC2MapReduceOutput)((ArrayList)node2).get(1));});*/


                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                final int tempCount = batchCount[ba];
                final double[] tempEtopStick = EtopStick;
                final double[][] tempElowerstick = Elowerstick;
                final double[] tempEsharestick = Esharestick;
                final HashMap<Integer, double[]> tempcachedContEloglik = cachedContEloglik;
              /*  MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions(
                        new FlatMapFunction<Iterator<MC2Data>,MC2MapReduceOutput>(){
                            @Override
                            public Iterable<MC2MapReduceOutput> call(Iterator<MC2Data>listDocs) throws Exception {
                                MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                                ArrayList output=new ArrayList();
                                output.add(temp);
                                return output;
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput ,MC2MapReduceOutput ,MC2MapReduceOutput>(){
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput  node2) throws Exception {
                                return reduceOutput(node1,node2);
                            }
                        }
                );*/
                final MC2Parameters params = new MC2Parameters(KK, TT, MM, aa, gg, ee, ngroups, qcc, q0, p0);


                MC2MapReduceOutput chunkOutput = (MC2MapReduceOutput) docChunk.map(
                        new Function<MC2Data, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2Data doc) throws Exception {
                                return computeParamEachCategoryDoc(doc, tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick, params);
//                                return computeParamEachCategoryDocFast(doc, tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick, params);
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput, MC2MapReduceOutput, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput node2) throws Exception {
                                return reduceOutput(node1, node2);
                            }
                        }
                );


                // Update global parameters
                double varpi = FastMath.pow(ba + 1 + (iter - 1) * nBatches + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT; tt++) {
                        for (int mm = 0; mm < MM - 1; mm++) {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chunkOutput.docChishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++) {
                            qcc[kk][tt][mm] = FastMath.exp(chis[kk][tt][mm] - maxChis[kk][tt]);
                            sum += qcc[kk][tt][mm];
                        }
                        qcc[kk][tt][MM - 1] = FastMath.exp(-maxChis[kk][tt]);
                        sum += qcc[kk][tt][MM - 1];
                        //Normalize
                        for (int mm = 0; mm < MM; mm++)
                            qcc[kk][tt][mm] = qcc[kk][tt][mm] / sum;
                    }
                }

                // Update \rhos
                for (int kk = 0; kk < KK - 1; kk++) {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT - 1; tt++) {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++) {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][1];
                }
                try {
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++) {
                        qq.get(kk).stochasticUpdate(chunkOutput.docAlphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++) {
                        pp.get(mm).stochasticUpdate(chunkOutput.docLambdahat.get(mm), varpi);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder + File.separator + "batch_" + ba + "iter" + iter + "_mc2_svi_spark.mat";
                System.out.println("\tExporting SVI output to Mat files...");
                try {
                    System.out.println("\t\t Data file: " + strOut);
                    MatlabJavaConverter.exportMC2SVIResultToMatWithQzz(KK, TT, MM, qcc, rhos, zetas, varphis, chunkOutput.qzz, ngroups, numdata, elapse, qq, pp, strOut, 1);
                    //   MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
    }

    /**
     * SVI inference for document of which data are categorical
     *
     * @param numIter      - the number of iterations (epoches)
     * @param batchSize    - mini batch size
     * @param varrho       - part of learning rate
     * @param iota         - part of learning rate (power)
     * @param strOutFolder - path to output folder
     */
    public static void sviCategoryOptimizerNaiveMF(int numIter, int batchSize, double varrho, double iota, String strOutFolder) {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++) {
            batchCount[i] = batchSize;
        }
        if ((ngroups - ngroups / batchSize * batchSize) > 0)

            batchCount[nBatches - 1] = ngroups - ngroups / batchSize * batchSize;
        else
            batchCount[nBatches - 1] = batchSize;

        Random rnd = new Random();
        rnd.setSeed(6789);
        /**
         * Start iterations
         */
        for (int iter = 1; iter <= numIter; iter++) {
            System.out.println("Iteration " + iter + " K=" + KK + " M=" + MM + " elapse=" + elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++) {
                timer.start();
                System.out.println("\t Running mini-batch " + ba + " from document # " + (noProcessedDocs + 1) + " to " + (noProcessedDocs + batchCount[ba]) + ", elapse= " + elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick = null;
                double[][] Elowerstick = null;
                double[] Esharestick = null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick = new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                HashMap<Integer, double[]> cachedContEloglik = new HashMap<Integer, double[]>();

                PerformanceTimer myTimer = new PerformanceTimer();
                //double total = 0;
                myTimer.start();
                final int startDoc = noProcessedDocs; //inclusive
                final int endDoc = noProcessedDocs + batchCount[ba]; //exclusive

               /* JavaRDD docChunk=corpus.filter(doc -> (doc.id>=startDoc && doc.id<endDoc));

                //Construct the key - word ids - for cachedContEloglik
                docChunk.collect().forEach((Object doc) -> {
                    ((MC2Data)doc).ss.forEach((Object word) -> {
                                if (!cachedContEloglik.containsKey((Integer) word)) {
                                    cachedContEloglik.put((Integer) word, null);
                                }
                            }
                    );
                });*/

                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                JavaRDD docChunk = corpus.filter(
                        new Function<MC2Data, Boolean>() {
                            @Override
                            public Boolean call(MC2Data doc) throws Exception {
                                return (doc.id >= startDoc && doc.id < endDoc);
                            }
                        }
                );
                //Construct the key - word ids - for cachedContEloglik
                for (Object doc : docChunk.collect()) {
                    for (Object word : ((MC2Data) doc).ss) {
                        if (!cachedContEloglik.containsKey((Integer) word)) {
                            cachedContEloglik.put((Integer) word, null);
                        }
                    }
                }

                myTimer.stop();
                System.out.println("\tRunning time for  getting word ids: " + myTimer.getElaspedSeconds());
                myTimer.start();
                ArrayList<Integer> keys = new ArrayList<Integer>(cachedContEloglik.keySet());
                double[][] tempEll = MultinomialDirichlet.expectationLogLikelihood(keys, pp);
                for (int ii = 0; ii < keys.size(); ii++)
                    cachedContEloglik.put(keys.get(ii), tempEll[ii]);
                myTimer.stop();
                System.out.println("\tRunning time for  computing expectation: " + myTimer.getElaspedSeconds());





               /*  int tempCount=batchCount[ba];
                double[] tempEtopStick =EtopStick;
                double[][] tempElowerstick =Elowerstick;
                double[] tempEsharestick = Esharestick;*/
                //int tempKK=KK;
                // int tempTT=TT;
                //  int tempMM=MM;


/*                MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions((listDocs)->{
                   //System.out.println(((MC2Data) doc).id);
                     // test();
                   // return null;
                   // return computeParamEachDoc((MC2Data)doc,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   ArrayList output=new ArrayList();
                   output.add(temp);
                   return output;
                }).reduce((node1, node2)->{return reduceOutput((MC2MapReduceOutput) ((ArrayList)node1).get(0),(MC2MapReduceOutput)((ArrayList)node2).get(1));});*/


                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                final int tempCount = batchCount[ba];
                final double[] tempEtopStick = EtopStick;
                final double[][] tempElowerstick = Elowerstick;
                final double[] tempEsharestick = Esharestick;
                final HashMap<Integer, double[]> tempcachedContEloglik = cachedContEloglik;
              /*  MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions(
                        new FlatMapFunction<Iterator<MC2Data>,MC2MapReduceOutput>(){
                            @Override
                            public Iterable<MC2MapReduceOutput> call(Iterator<MC2Data>listDocs) throws Exception {
                                MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                                ArrayList output=new ArrayList();
                                output.add(temp);
                                return output;
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput ,MC2MapReduceOutput ,MC2MapReduceOutput>(){
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput  node2) throws Exception {
                                return reduceOutput(node1,node2);
                            }
                        }
                );*/
                final MC2Parameters params = new MC2Parameters(KK, TT, MM, aa, gg, ee, ngroups, qcc, q0, p0);


                MC2MapReduceOutput chunkOutput = (MC2MapReduceOutput) docChunk.map(
                        new Function<MC2Data, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2Data doc) throws Exception {
                                return computeParamEachCategoryDocNaiveMF(doc, tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick, params);
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput, MC2MapReduceOutput, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput node2) throws Exception {
                                return reduceOutput(node1, node2);
                            }
                        }
                );


                // Update global parameters
                double varpi = FastMath.pow(ba + 1 + (iter - 1) * nBatches + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT; tt++) {
                        for (int mm = 0; mm < MM - 1; mm++) {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chunkOutput.docChishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++) {
                            qcc[kk][tt][mm] = FastMath.exp(chis[kk][tt][mm] - maxChis[kk][tt]);
                            sum += qcc[kk][tt][mm];
                        }
                        qcc[kk][tt][MM - 1] = FastMath.exp(-maxChis[kk][tt]);
                        sum += qcc[kk][tt][MM - 1];
                        //Normalize
                        for (int mm = 0; mm < MM; mm++)
                            qcc[kk][tt][mm] = qcc[kk][tt][mm] / sum;
                    }
                }

                // Update \rhos
                for (int kk = 0; kk < KK - 1; kk++) {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT - 1; tt++) {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++) {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][1];
                }
                try {
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++) {
                        qq.get(kk).stochasticUpdate(chunkOutput.docAlphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++) {
                        pp.get(mm).stochasticUpdate(chunkOutput.docLambdahat.get(mm), varpi);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder + File.separator + "batch_" + ba + "iter" + iter + "_mc2_svi_spark.mat";
                System.out.println("\tExporting SVI output to Mat files...");
                try {
                    System.out.println("\t\t Data file: " + strOut);
                    MatlabJavaConverter.exportMC2SVIResultToMatWithQzz(KK, TT, MM, qcc, rhos, zetas, varphis, chunkOutput.qzz, ngroups, numdata, elapse, qq, pp, strOut, 1);
                    //   MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
    }

    /**
     * SVI inference for document of which data are categorical
     *
     * @param numIter         - the number of iterations (epoches)
     * @param batchSize       - mini batch size
     * @param varrho          - part of learning rate
     * @param iota            - part of learning rate (power)
     * @param strOutFolder    - path to output folder
     * @param testFilePath    - path to the test file
     * @param batchResolution - the batch resolution for saving testing output
     */

    public static void sviCategoryOptimizer(int numIter, int batchSize, double varrho, double iota, String strOutFolder, String testFilePath, int batchResolution) {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++) {
            batchCount[i] = batchSize;
        }
        if ((ngroups - ngroups / batchSize * batchSize) > 0)

            batchCount[nBatches - 1] = ngroups - ngroups / batchSize * batchSize;
        else
            batchCount[nBatches - 1] = batchSize;

        Random rnd = new Random();
        rnd.setSeed(6789);
        /**
         * Start iterations
         */
        for (int iter = 1; iter <= numIter; iter++) {
            System.out.println("Iteration " + iter + " K=" + KK + " M=" + MM + " elapse=" + elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++) {
                timer.start();
                System.out.println("\t Running mini-batch " + ba + " from document # " + (noProcessedDocs + 1) + " to " + (noProcessedDocs + batchCount[ba]) + ", elapse= " + elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick = null;
                double[][] Elowerstick = null;
                double[] Esharestick = null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick = new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                HashMap<Integer, double[]> cachedContEloglik = new HashMap<Integer, double[]>();

                PerformanceTimer myTimer = new PerformanceTimer();
                //double total = 0;
                myTimer.start();
                final int startDoc = noProcessedDocs; //inclusive
                final int endDoc = noProcessedDocs + batchCount[ba]; //exclusive

               /* JavaRDD docChunk=corpus.filter(doc -> (doc.id>=startDoc && doc.id<endDoc));

                //Construct the key - word ids - for cachedContEloglik
                docChunk.collect().forEach((Object doc) -> {
                    ((MC2Data)doc).ss.forEach((Object word) -> {
                                if (!cachedContEloglik.containsKey((Integer) word)) {
                                    cachedContEloglik.put((Integer) word, null);
                                }
                            }
                    );
                });*/

                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                JavaRDD docChunk = corpus.filter(
                        new Function<MC2Data, Boolean>() {
                            @Override
                            public Boolean call(MC2Data doc) throws Exception {
                                return (doc.id >= startDoc && doc.id < endDoc);
                            }
                        }
                );
                //Construct the key - word ids - for cachedContEloglik
                for (Object doc : docChunk.collect()) {
                    for (Object word : ((MC2Data) doc).ss) {
                        if (!cachedContEloglik.containsKey((Integer) word)) {
                            cachedContEloglik.put((Integer) word, null);
                        }
                    }
                }

                myTimer.stop();
                System.out.println("\tRunning time for  getting word ids: " + myTimer.getElaspedSeconds());
                myTimer.start();
                ArrayList<Integer> keys = new ArrayList<Integer>(cachedContEloglik.keySet());
                double[][] tempEll = MultinomialDirichlet.expectationLogLikelihood(keys, pp);
                for (int ii = 0; ii < keys.size(); ii++)
                    cachedContEloglik.put(keys.get(ii), tempEll[ii]);
                myTimer.stop();
                System.out.println("\tRunning time for  computing expectation: " + myTimer.getElaspedSeconds());





               /*  int tempCount=batchCount[ba];
                double[] tempEtopStick =EtopStick;
                double[][] tempElowerstick =Elowerstick;
                double[] tempEsharestick = Esharestick;*/
                //int tempKK=KK;
                // int tempTT=TT;
                //  int tempMM=MM;


/*                MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions((listDocs)->{
                   //System.out.println(((MC2Data) doc).id);
                     // test();
                   // return null;
                   // return computeParamEachDoc((MC2Data)doc,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   ArrayList output=new ArrayList();
                   output.add(temp);
                   return output;
                }).reduce((node1, node2)->{return reduceOutput((MC2MapReduceOutput) ((ArrayList)node1).get(0),(MC2MapReduceOutput)((ArrayList)node2).get(1));});*/


                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                final int tempCount = batchCount[ba];
                final double[] tempEtopStick = EtopStick;
                final double[][] tempElowerstick = Elowerstick;
                final double[] tempEsharestick = Esharestick;
                final HashMap<Integer, double[]> tempcachedContEloglik = cachedContEloglik;
              /*  MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions(
                        new FlatMapFunction<Iterator<MC2Data>,MC2MapReduceOutput>(){
                            @Override
                            public Iterable<MC2MapReduceOutput> call(Iterator<MC2Data>listDocs) throws Exception {
                                MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                                ArrayList output=new ArrayList();
                                output.add(temp);
                                return output;
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput ,MC2MapReduceOutput ,MC2MapReduceOutput>(){
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput  node2) throws Exception {
                                return reduceOutput(node1,node2);
                            }
                        }
                );*/
                final MC2Parameters params = new MC2Parameters(KK, TT, MM, aa, gg, ee, ngroups, qcc, q0, p0);


                MC2MapReduceOutput chunkOutput = (MC2MapReduceOutput) docChunk.map(
                        new Function<MC2Data, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2Data doc) throws Exception {
                                return computeParamEachCategoryDoc(doc, tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick, params);
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput, MC2MapReduceOutput, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput node2) throws Exception {
                                return reduceOutput(node1, node2);
                            }
                        }
                );


                // Update global parameters
                double varpi = FastMath.pow(ba + 1 + (iter - 1) * nBatches + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT; tt++) {
                        for (int mm = 0; mm < MM - 1; mm++) {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chunkOutput.docChishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++) {
                            qcc[kk][tt][mm] = FastMath.exp(chis[kk][tt][mm] - maxChis[kk][tt]);
                            sum += qcc[kk][tt][mm];
                        }
                        qcc[kk][tt][MM - 1] = FastMath.exp(-maxChis[kk][tt]);
                        sum += qcc[kk][tt][MM - 1];
                        //Normalize
                        for (int mm = 0; mm < MM; mm++)
                            qcc[kk][tt][mm] = qcc[kk][tt][mm] / sum;
                    }
                }

                // Update \rhos
                for (int kk = 0; kk < KK - 1; kk++) {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT - 1; tt++) {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++) {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][1];
                }
                try {
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++) {
                        qq.get(kk).stochasticUpdate(chunkOutput.docAlphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++) {
                        pp.get(mm).stochasticUpdate(chunkOutput.docLambdahat.get(mm), varpi);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder + File.separator + "batch_" + ba + "iter" + iter + "_mc2_svi_spark.mat";
                System.out.println("\tExporting SVI output to Mat files...");
                try {
                    System.out.println("\t\t Data file: " + strOut);
                    MatlabJavaConverter.exportMC2SVIResultToMatWithQzz(KK, TT, MM, qcc, rhos, zetas, varphis, chunkOutput.qzz, ngroups, numdata, elapse, qq, pp, strOut, 1);
                    //   MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
    }

    public static MC2MapReduceOutput computeParamCategoryDocs(Iterator<MC2Data> docs, int batchCount, HashMap<Integer, double[]> cachedContEloglik,
                                                              double[] EtopStick, double[][] Elowerstick, double[] Esharestick, MC2Parameters params) {
        MC2MapReduceOutput result = new MC2MapReduceOutput();
        //Ouput variables
        result.docRhoshat = new double[KK - 1][2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();


        // Context
        for (int kk = 0; kk < KK; kk++) {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }
        int count = 0;
        while (docs.hasNext()) {
            result = reduceOutput(result, computeParamEachCategoryDoc(docs.next(), batchCount, cachedContEloglik, EtopStick, Elowerstick, Esharestick, params));
            count++;
        }
        System.out.println("Partition size: " + count);
        return result;
    }

    public static MC2MapReduceOutput computeParamEachCategoryDoc(MC2Data doc, int batchCount, HashMap<Integer, double[]> cachedContEloglik,
                                                                 double[] EtopStick, double[][] Elowerstick, double[] Esharestick,
                                                                 MC2Parameters params) {
        int KK = params.KK;
        int TT = params.TT;
        int MM = params.MM;
        BayesianComponent q0 = params.q0;
        BayesianComponent p0 = params.p0;
        double aa = params.aa;
        double gg = params.gg;
        double ee = params.ee;
        int ngroups = params.ngroups;
        double[][][] qcc = params.qcc;

        MC2MapReduceOutput result = new MC2MapReduceOutput();

        PerformanceTimer innerTimer = new PerformanceTimer();

        //Ouput variables
        result.docRhoshat = new double[KK - 1][2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();

        // Context
        for (int kk = 0; kk < KK; kk++) {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }

        int jj = doc.id;
        innerTimer.start();
        System.out.println("\t\t Running document # " + jj);

        // Computing expectation of log likelihood
        HashMap<Object, Integer> contMap = GetUniqueValue(doc.ss);
        ArrayList<Object> contData = new ArrayList<Object>(contMap.keySet());

        double[][] contEloglik = new double[contData.size()][MM];
        double[] cxtEloglik = new double[KK];
        try {
            for (int kk = 0; kk < KK; kk++) {
                cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(doc.xx);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (int ii = 0; ii < contData.size(); ii++) {
            //double[] vals =
            //for (int mm = 0; mm < MM; mm++) {
            contEloglik[ii] = cachedContEloglik.get(contData.get(ii));
            ;
            //}
        }

        //region Update local parameters \kappa,\vartheta

        //Computing \kappa

        double[][][] qtt = new double[contData.size()][KK][TT];
        double[][][] unormalizedQtt = new double[contData.size()][KK][TT]; // used for computing qzz
        double[][] maxQtt = new double[contData.size()][KK];
        double[][] sumUnormalizedQtt = new double[contData.size()][KK];

        //Compute qtt
        for (int ii = 0; ii < contData.size(); ii++) {
            for (int kk = 0; kk < KK; kk++) {
                maxQtt[ii][kk] = Double.NEGATIVE_INFINITY;

                for (int tt = 0; tt < TT; tt++) {
                    unormalizedQtt[ii][kk][tt] = 0;
                    for (int mm = 0; mm < MM; mm++) {
                        unormalizedQtt[ii][kk][tt] += qcc[kk][tt][mm] * contEloglik[ii][mm];
                    }
                    unormalizedQtt[ii][kk][tt] += Elowerstick[kk][tt];
                    if (unormalizedQtt[ii][kk][tt] > maxQtt[ii][kk])
                        maxQtt[ii][kk] = unormalizedQtt[ii][kk][tt];
                }
                // Convert to exponential
                sumUnormalizedQtt[ii][kk] = 0;
                for (int tt = 0; tt < TT; tt++) {
                    unormalizedQtt[ii][kk][tt] = FastMath.exp(unormalizedQtt[ii][kk][tt] - maxQtt[ii][kk]);
                    sumUnormalizedQtt[ii][kk] += unormalizedQtt[ii][kk][tt];
                }
                //Normalize
                for (int tt = 0; tt < TT; tt++)
                    qtt[ii][kk][tt] = unormalizedQtt[ii][kk][tt] / sumUnormalizedQtt[ii][kk];
            }
        }

        //Computing \vartheta
        double[] qzz = new double[KK];
        double maxQzz = Double.NEGATIVE_INFINITY;
        //Debug
        // System.out.println("qzz[k] \t EtopStick[kk] \t cxtEloglik[kk]");
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = 0;
            for (int ii = 0; ii < contData.size(); ii++) {
//                sum = 0;
//                for (int tt = 0; tt < TT; tt++)
//                    sum += unormalizedQtt[ii][kk][tt];
                qzz[kk] += (maxQtt[ii][kk] + FastMath.log(sumUnormalizedQtt[ii][kk])) * contMap.get(contData.get(ii));// multiply number of occurences of token
            }
// Debug
            //System.out.println(qzz[kk]+" \t "+EtopStick[kk]+" \t "+cxtEloglik[kk]);
            qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
            if (qzz[kk] > maxQzz)
                maxQzz = qzz[kk];
        }

        // Convert to expential
        double sum = 0;
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
            sum += qzz[kk];
        }
        //Normalize
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = qzz[kk] / sum;
        }

        // For computing clustering performance
        result.qzz = new HashMap();
        result.qzz.put(doc.id, qzz);


        //region Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat


        //Update \chihat,\zetashat,\lambdashat
        double[][][] alphas = new double[KK][TT][MM];
        double[][] dataProb = new double[contData.size()][MM];


        for (int kk = 0; kk < KK; kk++) {

            // Computing the last TT
            double[] sumzetas = new double[TT];
            for (int tt = 0; tt < TT; tt++) {
                sum = 0;
                //Computing alphas(:,:,_MM-1)
                for (int ii = 0; ii < contData.size(); ii++) {
                    sum += qtt[ii][kk][tt] * contEloglik[ii][MM - 1] * contMap.get(contData.get(ii)); // multiply number of occurences of token
                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
                    for (int mm = 0; mm < MM; mm++) {
                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
                    }

                }
                alphas[kk][tt][MM - 1] = qzz[kk] * sum;


                //Computing alphas(:,:,0:_MM-2) and chishat
                for (int mm = 0; mm < MM - 1; mm++) {
                    sum = 0;
                    for (int ii = 0; ii < contData.size(); ii++)
                        sum += qtt[ii][kk][tt] * contEloglik[ii][mm] * contMap.get(contData.get(ii)); // multiply number of occurences of token;
                    alphas[kk][tt][mm] = qzz[kk] * sum;
                    result.docChishat[kk][tt][mm] = Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                }
            }
            sum = sumzetas[TT - 1];
            for (int tt = TT - 2; tt >= 0; tt--) {
                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
                sum += sumzetas[tt];

            }
        }


        //Update \rhoshat
        sum = qzz[KK - 1];
        for (int kk = KK - 2; kk >= 0; kk--) {
            result.docRhoshat[kk][0] = 1 + ngroups * qzz[kk];
            result.docRhoshat[kk][1] = gg + ngroups * sum;
            sum += qzz[kk];

        }

        //Update \zetashat
//        for (int kk = 0; kk < KK; kk++) {
//            // Computing the last TT
//            double[] sumzetas = new double[TT];
//            for (int ii = 0; ii < contData.size(); ii++)
//                for (int tt = 0; tt < TT; tt++) {
//                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
//                }
//            sum = sumzetas[TT - 1];
//            for (int tt = TT - 2; tt >= 0; tt--) {
//                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
//                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
//                sum += sumzetas[tt];
//
//            }
//        }

        //Update \varphishat
        double[] sumkt = new double[MM];

        for (int mm = 0; mm < MM; mm++) {
            for (int kk = 0; kk < KK; kk++) {
                for (int tt = 0; tt < TT; tt++) {
                    sumkt[mm] += qcc[kk][tt][mm];
                }
            }
        }
        sum = sumkt[MM - 1];
        for (int mm = MM - 2; mm >= 0; mm--) {

//            result.docVarphishat[mm][0] = 1 + ngroups * sumkt[mm];
//            result.docVarphishat[mm][1] = ee + ngroups * sum;
            result.docVarphishat[mm][0] = 1 + sumkt[mm];
            result.docVarphishat[mm][1] = ee + sum;
            sum += sumkt[mm];

        }

        // Update \lambdashat
//        double[][] dataProb = new double[contData.size()][MM];
        for (int ii = 0; ii < contData.size(); ii++)
            for (int mm = 0; mm < MM; mm++) {
//                for (int kk = 0; kk < KK; kk++)
//                    for (int tt = 0; tt < TT; tt++) {
//                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
//                    }
                try {
                    result.docLambdahat.get(mm).add(contData.get(ii), ngroups / batchCount * dataProb[ii][mm] * contMap.get(contData.get(ii))); // multiply number of occurences of token
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        //Updating \alphashat
        try {
            for (int kk = 0; kk < KK; kk++) {
                result.docAlphahat.get(kk).add(doc.xx, ngroups / batchCount * qzz[kk]);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        // This is new command to save qzz for computing clustering performance
//        result.qzz= new double[1][qzz.length];
//        for (int kk=0;kk<qzz.length;kk++)
//        result.qzz[0][kk]=qzz[kk];
        // Computing running time
        innerTimer.stop();
        System.out.println("\t\tFinished document " + jj + " elapse=" + innerTimer.getElaspedSeconds());
        return result;
    }

    public static MC2MapReduceOutput computeParamEachCategoryDocFast(MC2Data doc, int batchCount, HashMap<Integer, double[]> cachedContEloglik,
                                                                     double[] EtopStick, double[][] Elowerstick, double[] Esharestick,
                                                                     MC2Parameters params) {
        int KK = params.KK;
        int TT = params.TT;
        int MM = params.MM;
        BayesianComponent q0 = params.q0;
        BayesianComponent p0 = params.p0;
        double aa = params.aa;
        double gg = params.gg;
        double ee = params.ee;
        int ngroups = params.ngroups;
        double[][][] qcc = params.qcc;

        MC2MapReduceOutput result = new MC2MapReduceOutput();

        PerformanceTimer innerTimer = new PerformanceTimer();

        //Ouput variables
        result.docRhoshat = new double[KK - 1][2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();

        // Context
        for (int kk = 0; kk < KK; kk++) {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }

        int jj = doc.id;
        innerTimer.start();
        System.out.println("\t\t Running document # " + jj);

        // Computing expectation of log likelihood
        HashMap<Object, Integer> contMap = GetUniqueValue(doc.ss);
        ArrayList<Object> contData = new ArrayList<Object>(contMap.keySet());

        double[][] contEloglik = new double[contData.size()][MM];
        double[] cxtEloglik = new double[KK];
        try {
            for (int kk = 0; kk < KK; kk++) {
                cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(doc.xx);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (int ii = 0; ii < contData.size(); ii++) {
            //double[] vals =
            //for (int mm = 0; mm < MM; mm++) {
            contEloglik[ii] = cachedContEloglik.get(contData.get(ii));
            ;
            //}
        }

        //region Update local parameters \kappa,\vartheta

        //Computing \kappa

        double[][][] qtt = new double[contData.size()][KK][TT];
        double[][][] unormalizedQtt = new double[contData.size()][KK][TT]; // used for computing qzz
        double[][] maxQtt = new double[contData.size()][KK];
        double[][] sumUnormalizedQtt = new double[contData.size()][KK];
        unormalizedQtt = FastMatrixFunctions.multNDArrayJBLAS(qcc, contEloglik, Elowerstick);
        //Compute qtt
        for (int ii = 0; ii < contData.size(); ii++) {
//            for (int kk = 0; kk < KK; kk++) {
            DoubleMatrix tempMaxQtt = (new DoubleMatrix(unormalizedQtt[ii])).rowMaxs();

//            }
            DoubleMatrix temp = new DoubleMatrix(unormalizedQtt[ii]);
            temp = MatrixFunctions.exp(temp.subColumnVector(tempMaxQtt));
            unormalizedQtt[ii] = temp.toArray2();
            qtt[ii] = temp.divColumnVector(temp.rowSums()).toArray2();
//                // Convert to exponential
//                sumUnormalizedQtt[ii][kk] = 0;
//                for (int tt = 0; tt < TT; tt++) {
//                    unormalizedQtt[ii][kk][tt] = FastMath.exp(unormalizedQtt[ii][kk][tt] - maxQtt[ii][kk]);
//                    sumUnormalizedQtt[ii][kk] += unormalizedQtt[ii][kk][tt];
//                }
//                //Normalize
//                for (int tt = 0; tt < TT; tt++)
//                    qtt[ii][kk][tt] = unormalizedQtt[ii][kk][tt] / sumUnormalizedQtt[ii][kk];
//            }
            maxQtt[ii] = tempMaxQtt.toArray();
        }

        //Computing \vartheta
        double[] qzz = new double[KK];
        double maxQzz = Double.NEGATIVE_INFINITY;
        //Debug
        // System.out.println("qzz[k] \t EtopStick[kk] \t cxtEloglik[kk]");
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = 0;
            for (int ii = 0; ii < contData.size(); ii++) {
//                sum = 0;
//                for (int tt = 0; tt < TT; tt++)
//                    sum += unormalizedQtt[ii][kk][tt];
                qzz[kk] += (maxQtt[ii][kk] + FastMath.log(sumUnormalizedQtt[ii][kk])) * contMap.get(contData.get(ii));// multiply number of occurences of token
            }
// Debug
            //System.out.println(qzz[kk]+" \t "+EtopStick[kk]+" \t "+cxtEloglik[kk]);
            qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
            if (qzz[kk] > maxQzz)
                maxQzz = qzz[kk];
        }

        // Convert to expential
        double sum = 0;
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
            sum += qzz[kk];
        }
        //Normalize
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = qzz[kk] / sum;
        }

        // For computing clustering performance
        result.qzz = new HashMap();
        result.qzz.put(doc.id, qzz);


        //region Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat


        //Update \chihat,\zetashat,\lambdashat
        double[][][] alphas = new double[KK][TT][MM];
        double[][] dataProb = new double[contData.size()][MM];

        dataProb = FastMatrixFunctions.multNDArrayJBLAS(qtt, qcc, qzz);
        for (int kk = 0; kk < KK; kk++) {

            // Computing the last TT
            double[] sumzetas = new double[TT];
            for (int tt = 0; tt < TT; tt++) {
                sum = 0;
                //Computing alphas(:,:,_MM-1)
                for (int ii = 0; ii < contData.size(); ii++) {
                    sum += qtt[ii][kk][tt] * contEloglik[ii][MM - 1] * contMap.get(contData.get(ii)); // multiply number of occurences of token
                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
//                    for (int mm = 0; mm < MM; mm++) {
//                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
//                    }

                }
                alphas[kk][tt][MM - 1] = qzz[kk] * sum;


                //Computing alphas(:,:,0:_MM-2) and chishat
                for (int mm = 0; mm < MM - 1; mm++) {
                    sum = 0;
                    for (int ii = 0; ii < contData.size(); ii++)
                        sum += qtt[ii][kk][tt] * contEloglik[ii][mm] * contMap.get(contData.get(ii)); // multiply number of occurences of token;
                    alphas[kk][tt][mm] = qzz[kk] * sum;
                    result.docChishat[kk][tt][mm] = Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                }
            }
            sum = sumzetas[TT - 1];
            for (int tt = TT - 2; tt >= 0; tt--) {
                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
                sum += sumzetas[tt];

            }
        }


        //Update \rhoshat
        sum = qzz[KK - 1];
        for (int kk = KK - 2; kk >= 0; kk--) {
            result.docRhoshat[kk][0] = 1 + ngroups * qzz[kk];
            result.docRhoshat[kk][1] = gg + ngroups * sum;
            sum += qzz[kk];

        }

        //Update \zetashat
//        for (int kk = 0; kk < KK; kk++) {
//            // Computing the last TT
//            double[] sumzetas = new double[TT];
//            for (int ii = 0; ii < contData.size(); ii++)
//                for (int tt = 0; tt < TT; tt++) {
//                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
//                }
//            sum = sumzetas[TT - 1];
//            for (int tt = TT - 2; tt >= 0; tt--) {
//                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
//                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
//                sum += sumzetas[tt];
//
//            }
//        }

        //Update \varphishat
        double[] sumkt = new double[MM];

        for (int mm = 0; mm < MM; mm++) {
            for (int kk = 0; kk < KK; kk++) {
                for (int tt = 0; tt < TT; tt++) {
                    sumkt[mm] += qcc[kk][tt][mm];
                }
            }
        }
        sum = sumkt[MM - 1];
        for (int mm = MM - 2; mm >= 0; mm--) {

//            result.docVarphishat[mm][0] = 1 + ngroups * sumkt[mm];
//            result.docVarphishat[mm][1] = ee + ngroups * sum;
            result.docVarphishat[mm][0] = 1 + sumkt[mm];
            result.docVarphishat[mm][1] = ee + sum;
            sum += sumkt[mm];

        }

        // Update \lambdashat
//        double[][] dataProb = new double[contData.size()][MM];
        for (int ii = 0; ii < contData.size(); ii++)
            for (int mm = 0; mm < MM; mm++) {
//                for (int kk = 0; kk < KK; kk++)
//                    for (int tt = 0; tt < TT; tt++) {
//                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
//                    }
                try {
                    result.docLambdahat.get(mm).add(contData.get(ii), ngroups / batchCount * dataProb[ii][mm] * contMap.get(contData.get(ii))); // multiply number of occurences of token
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        //Updating \alphashat
        try {
            for (int kk = 0; kk < KK; kk++) {
                result.docAlphahat.get(kk).add(doc.xx, ngroups / batchCount * qzz[kk]);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        // This is new command to save qzz for computing clustering performance
//        result.qzz= new double[1][qzz.length];
//        for (int kk=0;kk<qzz.length;kk++)
//        result.qzz[0][kk]=qzz[kk];
        // Computing running time
        innerTimer.stop();
        System.out.println("\t\tFinished document " + jj + " elapse=" + innerTimer.getElaspedSeconds());
        return result;
    }

    public static MC2MapReduceOutput computeParamEachCategoryDocNaiveMF(MC2Data doc, int batchCount, HashMap<Integer, double[]> cachedContEloglik,
                                                                        double[] EtopStick, double[][] Elowerstick, double[] Esharestick,
                                                                        MC2Parameters params) {
        int KK = params.KK;
        int TT = params.TT;
        int MM = params.MM;
        BayesianComponent q0 = params.q0;
        BayesianComponent p0 = params.p0;
        double aa = params.aa;
        double gg = params.gg;
        double ee = params.ee;
        int ngroups = params.ngroups;
        double[][][] qcc = params.qcc;

        MC2MapReduceOutput result = new MC2MapReduceOutput();

        PerformanceTimer innerTimer = new PerformanceTimer();

        //Ouput variables
        result.docRhoshat = new double[KK - 1][2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();

        // Context
        for (int kk = 0; kk < KK; kk++) {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }

        int jj = doc.id;
        innerTimer.start();
        System.out.println("\t\t Running document # " + jj);

        // Computing expectation of log likelihood
        HashMap<Object, Integer> contMap = GetUniqueValue(doc.ss);
        ArrayList<Object> contData = new ArrayList<Object>(contMap.keySet());

        double[][] contEloglik = new double[contData.size()][MM];
        double[] cxtEloglik = new double[KK];
        try {
            for (int kk = 0; kk < KK; kk++) {
                cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(doc.xx);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (int ii = 0; ii < contData.size(); ii++) {
            //double[] vals =
            //for (int mm = 0; mm < MM; mm++) {
            contEloglik[ii] = cachedContEloglik.get(contData.get(ii));
            ;
            //}
        }

        //region Update local parameters \kappa,\vartheta

        //Computing \kappa
        //Initilize qzz
        Random rnd = new Random();
        rnd.setSeed(6789);
        double[] qzz = new double[KK];
        double sum = 0;
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = rnd.nextDouble();
            sum += qzz[kk];
        }
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = qzz[kk] / sum;
        }

        double[][] qtt = new double[contData.size()][TT];
        double[][][] unormalizedQtt = new double[contData.size()][KK][TT]; // used for computing qzz
        double[] maxQtt = new double[contData.size()];
        double[] sumUnormalizedQtt = new double[contData.size()];
        for (int ii = 0; ii < contData.size(); ii++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int kk = 0; kk < KK; kk++) {
                    unormalizedQtt[ii][kk][tt] = 0;
                    for (int mm = 0; mm < MM; mm++) {
                        unormalizedQtt[ii][kk][tt] += qcc[kk][tt][mm] * contEloglik[ii][mm];
                    }
                    unormalizedQtt[ii][kk][tt] += Elowerstick[kk][tt];
                }
            }
        }
        int noInnIter = 3;

        for (int innIter = 0; innIter < noInnIter; innIter++) {
            //Compute qtt
            for (int ii = 0; ii < contData.size(); ii++) {
                for (int tt = 0; tt < TT; tt++) {
                    for (int kk = 0; kk < KK; kk++) {
                        qtt[ii][tt] += unormalizedQtt[ii][kk][tt] * qzz[kk];
                    }
                    maxQtt[ii] = Double.NEGATIVE_INFINITY;
                    if (qtt[ii][tt] > maxQtt[ii])
                        maxQtt[ii] = qtt[ii][tt];
                }
                // Convert to exponential
                sumUnormalizedQtt[ii] = 0;
                for (int tt = 0; tt < TT; tt++) {
                    qtt[ii][tt] = FastMath.exp(qtt[ii][tt] - maxQtt[ii]);
                    sumUnormalizedQtt[ii] += qtt[ii][tt];
                }
                //Normalize
                for (int tt = 0; tt < TT; tt++)
                    qtt[ii][tt] = qtt[ii][tt] / sumUnormalizedQtt[ii];
            }


            //Computing \vartheta

            double maxQzz = Double.NEGATIVE_INFINITY;
            //Debug
            // System.out.println("qzz[k] \t EtopStick[kk] \t cxtEloglik[kk]");
            for (int kk = 0; kk < KK; kk++) {
                qzz[kk] = 0;
                for (int ii = 0; ii < contData.size(); ii++) {
                    for (int tt = 0; tt < TT; tt++) {
                        qzz[kk] += qtt[ii][tt] * unormalizedQtt[ii][kk][tt] * contMap.get(contData.get(ii));// multiply number of occurences of token
                    }
                }
                // Debug
                //System.out.println(qzz[kk]+" \t "+EtopStick[kk]+" \t "+cxtEloglik[kk]);
                qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
                if (qzz[kk] > maxQzz)
                    maxQzz = qzz[kk];
            }


            // Convert to expential
            sum = 0;
            for (int kk = 0; kk < KK; kk++) {
                qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
                sum += qzz[kk];
            }
            //Normalize
            for (int kk = 0; kk < KK; kk++) {
                qzz[kk] = qzz[kk] / sum;
            }
        }
        // For computing clustering performance
        result.qzz = new HashMap();
        result.qzz.put(doc.id, qzz);


        //region Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat


        //Update \chihat,\zetashat,\lambdashat
        double[][][] alphas = new double[KK][TT][MM];
        double[][] dataProb = new double[contData.size()][MM];


        for (int kk = 0; kk < KK; kk++) {
            // Computing the last TT
            double[] sumzetas = new double[TT];
            for (int tt = 0; tt < TT; tt++) {
                sum = 0;
                //Computing alphas(:,:,MM-1)
                for (int ii = 0; ii < contData.size(); ii++) {
                    sum += qtt[ii][tt] * contEloglik[ii][MM - 1] * contMap.get(contData.get(ii)); // multiply number of occurrences of token
                    sumzetas[tt] += qtt[ii][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
                    for (int mm = 0; mm < MM; mm++) {
                        dataProb[ii][mm] += qtt[ii][tt] * qcc[kk][tt][mm] * qzz[kk];
                    }

                }
                alphas[kk][tt][MM - 1] = qzz[kk] * sum;


                //Computing alphas(:,:,0:_MM-2) and chishat
                for (int mm = 0; mm < MM - 1; mm++) {
                    sum = 0;
                    for (int ii = 0; ii < contData.size(); ii++)
                        sum += qtt[ii][tt] * contEloglik[ii][mm] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;
                    alphas[kk][tt][mm] = qzz[kk] * sum;
                    result.docChishat[kk][tt][mm] = Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                }
            }
            sum = sumzetas[TT - 1];
            for (int tt = TT - 2; tt >= 0; tt--) {
                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
                sum += sumzetas[tt];

            }
        }


        //Update \rhoshat
        sum = qzz[KK - 1];
        for (int kk = KK - 2; kk >= 0; kk--) {
            result.docRhoshat[kk][0] = 1 + ngroups * qzz[kk];
            result.docRhoshat[kk][1] = gg + ngroups * sum;
            sum += qzz[kk];

        }

        //Update \zetashat
//        for (int kk = 0; kk < KK; kk++) {
//            // Computing the last TT
//            double[] sumzetas = new double[TT];
//            for (int ii = 0; ii < contData.size(); ii++)
//                for (int tt = 0; tt < TT; tt++) {
//                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurrences of token;;
//                }
//            sum = sumzetas[TT - 1];
//            for (int tt = TT - 2; tt >= 0; tt--) {
//                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
//                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
//                sum += sumzetas[tt];
//
//            }
//        }

        //Update \varphishat
        double[] sumkt = new double[MM];

        for (int mm = 0; mm < MM; mm++) {
            for (int kk = 0; kk < KK; kk++) {
                for (int tt = 0; tt < TT; tt++) {
                    sumkt[mm] += qcc[kk][tt][mm];
                }
            }
        }
        sum = sumkt[MM - 1];
        for (int mm = MM - 2; mm >= 0; mm--) {

//            result.docVarphishat[mm][0] = 1 + ngroups * sumkt[mm];
//            result.docVarphishat[mm][1] = ee + ngroups * sum;
            result.docVarphishat[mm][0] = 1 + sumkt[mm];
            result.docVarphishat[mm][1] = ee + sum;
            sum += sumkt[mm];

        }

        // Update \lambdashat
//        double[][] dataProb = new double[contData.size()][MM];
        for (int ii = 0; ii < contData.size(); ii++)
            for (int mm = 0; mm < MM; mm++) {
//                for (int kk = 0; kk < KK; kk++)
//                    for (int tt = 0; tt < TT; tt++) {
//                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
//                    }
                try {
                    result.docLambdahat.get(mm).add(contData.get(ii), ngroups / batchCount * dataProb[ii][mm] * contMap.get(contData.get(ii))); // multiply number of occurrences of token
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        //Updating \alphashat
        try {
            for (int kk = 0; kk < KK; kk++) {
                result.docAlphahat.get(kk).add(doc.xx, ngroups / batchCount * qzz[kk]);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        // This is new command to save qzz for computing clustering performance
//        result.qzz= new double[1][qzz.length];
//        for (int kk=0;kk<qzz.length;kk++)
//        result.qzz[0][kk]=qzz[kk];
        // Computing running time
        innerTimer.stop();
        System.out.println("\t\tFinished document " + jj + " elapse=" + innerTimer.getElaspedSeconds());
        return result;
    }

    /**
     * SVI inference for document of which data are generic. When the data point is categorical, use sviCategoryOptimizer instead
     *
     * @param numIter      - the number of iterations (epoches)
     * @param batchSize    - mini batch size
     * @param varrho       - part of learning rate
     * @param iota         - part of learning rate (power)
     * @param strOutFolder - path to output folder
     */
    public static void sviOptimizer(int numIter, int batchSize, double varrho, double iota, String strOutFolder) {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++) {
            batchCount[i] = batchSize;
        }
        if ((ngroups - ngroups / batchSize * batchSize) > 0)

            batchCount[nBatches - 1] = ngroups - ngroups / batchSize * batchSize;
        else
            batchCount[nBatches - 1] = batchSize;

        Random rnd = new Random();
        rnd.setSeed(6789);
        /**
         * Start iterations
         */
        for (int iter = 1; iter <= numIter; iter++) {
            System.out.println("Iteration " + iter + " K=" + KK + " M=" + MM + " elapse=" + elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++) {
                timer.start();
                System.out.println("\t Running mini-batch " + ba + " from document # " + (noProcessedDocs + 1) + " to " + (noProcessedDocs + batchCount[ba]) + ", elapse= " + elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick = null;
                double[][] Elowerstick = null;
                double[] Esharestick = null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick = new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                //HashMap<Integer, double[]> cachedContEloglik = new HashMap<Integer, double[]>();

                PerformanceTimer myTimer = new PerformanceTimer();
                //double total = 0;
                myTimer.start();
                final int startDoc = noProcessedDocs; //inclusive
                final int endDoc = noProcessedDocs + batchCount[ba]; //exclusive

               /* JavaRDD docChunk=corpus.filter(doc -> (doc.id>=startDoc && doc.id<endDoc));

                //Construct the key - word ids - for cachedContEloglik
                docChunk.collect().forEach((Object doc) -> {
                    ((MC2Data)doc).ss.forEach((Object word) -> {
                                if (!cachedContEloglik.containsKey((Integer) word)) {
                                    cachedContEloglik.put((Integer) word, null);
                                }
                            }
                    );
                });*/

                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                JavaRDD docChunk = corpus.filter(
                        new Function<MC2Data, Boolean>() {
                            @Override
                            public Boolean call(MC2Data doc) throws Exception {
                                return (doc.id >= startDoc && doc.id < endDoc);
                            }
                        }
                );
                /*
                //Construct the key - word ids - for cachedContEloglik
                for(Object doc:docChunk.collect()){
                    for(Object word:((MC2Data)doc).ss) {
                        if (!cachedContEloglik.containsKey((Integer) word)) {
                            cachedContEloglik.put((Integer) word, null);
                        }
                    }
                }

                myTimer.stop();
                System.out.println("\tRunning time for  getting word ids: " + myTimer.getElaspedSeconds());
                myTimer.start();
                ArrayList<Integer> keys = new ArrayList<Integer>(cachedContEloglik.keySet());
                double[][] tempEll = MultinomialDirichlet.expectationLogLikelihood(keys, pp);
                for (int ii = 0; ii < keys.size(); ii++)
                    cachedContEloglik.put(keys.get(ii), tempEll[ii]);
                myTimer.stop();
                System.out.println("\tRunning time for  computing expectation: " + myTimer.getElaspedSeconds());
                */




               /*  int tempCount=batchCount[ba];
                double[] tempEtopStick =EtopStick;
                double[][] tempElowerstick =Elowerstick;
                double[] tempEsharestick = Esharestick;*/
                //int tempKK=KK;
                // int tempTT=TT;
                //  int tempMM=MM;


/*                MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions((listDocs)->{
                   //System.out.println(((MC2Data) doc).id);
                     // test();
                   // return null;
                   // return computeParamEachDoc((MC2Data)doc,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, cachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                   ArrayList output=new ArrayList();
                   output.add(temp);
                   return output;
                }).reduce((node1, node2)->{return reduceOutput((MC2MapReduceOutput) ((ArrayList)node1).get(0),(MC2MapReduceOutput)((ArrayList)node2).get(1));});*/


                //Replace lambda expression with version 7 - comment this and comment previous for using lamda expression
                final int tempCount = batchCount[ba];
                final double[] tempEtopStick = EtopStick;
                final double[][] tempElowerstick = Elowerstick;
                final double[] tempEsharestick = Esharestick;
                //  final   HashMap<Integer, double[]> tempcachedContEloglik = cachedContEloglik;
              /*  MC2MapReduceOutput chunkOutput=(MC2MapReduceOutput)docChunk.mapPartitions(
                        new FlatMapFunction<Iterator<MC2Data>,MC2MapReduceOutput>(){
                            @Override
                            public Iterable<MC2MapReduceOutput> call(Iterator<MC2Data>listDocs) throws Exception {
                                MC2MapReduceOutput temp=computeParamDocs((Iterator<MC2Data>) listDocs,tempCount, tempcachedContEloglik, tempEtopStick, tempElowerstick, tempEsharestick);
                                ArrayList output=new ArrayList();
                                output.add(temp);
                                return output;
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput ,MC2MapReduceOutput ,MC2MapReduceOutput>(){
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput  node2) throws Exception {
                                return reduceOutput(node1,node2);
                            }
                        }
                );*/
                final MC2Parameters params = new MC2Parameters(KK, TT, MM, aa, gg, ee, ngroups, qcc, q0, p0);

                List<MC2MapReduceOutput> test = docChunk.map(
                        new Function<MC2Data, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2Data doc) throws Exception {
                                return computeParamEachDoc(doc, tempCount, tempEtopStick, tempElowerstick, tempEsharestick, params);
                            }
                        }
                ).collect();
                MC2MapReduceOutput chunkOutput = (MC2MapReduceOutput) docChunk.map(
                        new Function<MC2Data, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2Data doc) throws Exception {
                                return computeParamEachDoc(doc, tempCount, tempEtopStick, tempElowerstick, tempEsharestick, params);
                            }
                        }
                ).reduce(
                        new Function2<MC2MapReduceOutput, MC2MapReduceOutput, MC2MapReduceOutput>() {
                            @Override
                            public MC2MapReduceOutput call(MC2MapReduceOutput node1, MC2MapReduceOutput node2) throws Exception {
                                return reduceOutput(node1, node2);
                            }
                        }
                );


                // Update global parameters
                double varpi = FastMath.pow(ba + 1 + (iter - 1) * nBatches + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT; tt++) {
                        for (int mm = 0; mm < MM - 1; mm++) {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chunkOutput.docChishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++) {
                            qcc[kk][tt][mm] = FastMath.exp(chis[kk][tt][mm] - maxChis[kk][tt]);
                            sum += qcc[kk][tt][mm];
                        }
                        qcc[kk][tt][MM - 1] = FastMath.exp(-maxChis[kk][tt]);
                        sum += qcc[kk][tt][MM - 1];
                        //Normalize
                        for (int mm = 0; mm < MM; mm++)
                            qcc[kk][tt][mm] = qcc[kk][tt][mm] / sum;
                    }
                }

                // Update \rhos
                for (int kk = 0; kk < KK - 1; kk++) {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * chunkOutput.docRhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++) {
                    for (int tt = 0; tt < TT - 1; tt++) {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * chunkOutput.docZetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++) {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * chunkOutput.docVarphishat[mm][1];
                }
                try {
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++) {
                        qq.get(kk).stochasticUpdate(chunkOutput.docAlphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++) {
                        pp.get(mm).stochasticUpdate(chunkOutput.docLambdahat.get(mm), varpi);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder + File.separator + "batch_" + ba + "iter" + iter + "_mc2_svi_spark.mat";
                System.out.println("\tExporting SVI output to Mat files...");
                try {
                    System.out.println("\t\t Data file: " + strOut);
                    MatlabJavaConverter.exportMC2SVIResultToMatWithQzz(KK, TT, MM, qcc, rhos, zetas, varphis, chunkOutput.qzz, ngroups, numdata, elapse, qq, pp, strOut, 1);
//                    MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
    }

    public static MC2MapReduceOutput computeParamEachDoc(MC2Data doc, int batchCount, double[] EtopStick, double[][] Elowerstick,
                                                         double[] Esharestick, MC2Parameters params) {
        int KK = params.KK;
        int TT = params.TT;
        int MM = params.MM;
        BayesianComponent q0 = params.q0;
        BayesianComponent p0 = params.p0;
        double aa = params.aa;
        double gg = params.gg;
        double ee = params.ee;
        int ngroups = params.ngroups;
        double[][][] qcc = params.qcc;

        MC2MapReduceOutput result = new MC2MapReduceOutput();

        PerformanceTimer innerTimer = new PerformanceTimer();

        //Ouput variables
        result.docRhoshat = new double[KK - 1][2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();


        // Context
        for (int kk = 0; kk < KK; kk++) {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++) {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }

        int jj = doc.id;
        innerTimer.start();
        System.out.println("\t\t Running document # " + jj);

        // Computing expectation of log likelihood
        //HashMap<Object, Integer> contMap = GetUniqueValue(doc.ss);
        // ArrayList<Object> contData = new ArrayList<Object>(contMap.keySet());

        double[][] contEloglik = new double[doc.ss.size()][MM];
        double[] cxtEloglik = new double[KK];
        try {
            for (int kk = 0; kk < KK; kk++) {
                cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(doc.xx);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            for (int ii = 0; ii < doc.ss.size(); ii++) {
                //double[] vals =
                for (int mm = 0; mm < MM; mm++) {
                    contEloglik[ii][mm] = pp.get(mm).expectationLogLikelihood(doc.ss.get(ii));
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        //region Update local parameters \kappa,\vartheta

        //Computing \kappa

        double[][][] qtt = new double[doc.ss.size()][KK][TT];
        double[][][] unormalizedQtt = new double[doc.ss.size()][KK][TT]; // used for computing qzz
        double[][] maxQtt = new double[doc.ss.size()][KK];
        double[][] sumUnormalizedQtt = new double[doc.ss.size()][KK];

        //Compute qtt
        for (int ii = 0; ii < doc.ss.size(); ii++) {
            for (int kk = 0; kk < KK; kk++) {
                maxQtt[ii][kk] = Double.NEGATIVE_INFINITY;

                for (int tt = 0; tt < TT; tt++) {
                    unormalizedQtt[ii][kk][tt] = 0;
                    for (int mm = 0; mm < MM; mm++) {
                        unormalizedQtt[ii][kk][tt] += qcc[kk][tt][mm] * contEloglik[ii][mm];
                    }
                    unormalizedQtt[ii][kk][tt] += Elowerstick[kk][tt];
                    if (unormalizedQtt[ii][kk][tt] > maxQtt[ii][kk])
                        maxQtt[ii][kk] = unormalizedQtt[ii][kk][tt];
                }
                // Convert to exponential
                sumUnormalizedQtt[ii][kk] = 0;
                for (int tt = 0; tt < TT; tt++) {
                    unormalizedQtt[ii][kk][tt] = FastMath.exp(unormalizedQtt[ii][kk][tt] - maxQtt[ii][kk]);
                    sumUnormalizedQtt[ii][kk] += unormalizedQtt[ii][kk][tt];
                }
                //Normalize
                for (int tt = 0; tt < TT; tt++)
                    qtt[ii][kk][tt] = unormalizedQtt[ii][kk][tt] / sumUnormalizedQtt[ii][kk];
            }
        }

        //Computing \vartheta
        double[] qzz = new double[KK];
        double maxQzz = Double.NEGATIVE_INFINITY;
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = 0;
            for (int ii = 0; ii < doc.ss.size(); ii++) {
//                sum = 0;
//                for (int tt = 0; tt < TT; tt++)
//                    sum += unormalizedQtt[ii][kk][tt];
                qzz[kk] += (maxQtt[ii][kk] + FastMath.log(sumUnormalizedQtt[ii][kk]));// multiply number of occurrences of token
            }
            if (qzz[kk] > maxQzz)
                maxQzz = qzz[kk];
            qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
        }

        // Convert to expential
        double sum = 0;
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
            sum += qzz[kk];
        }
        //Normalize
        for (int kk = 0; kk < KK; kk++) {
            qzz[kk] = qzz[kk] / sum;
        }


        //region Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat


        //Update \chihat
        double[][][] alphas = new double[KK][TT][MM];

        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                sum = 0;
                //Computing alphas(:,:,_MM-1)
                for (int ii = 0; ii < doc.ss.size(); ii++)
                    sum += qtt[ii][kk][tt] * contEloglik[ii][MM - 1]; // multiply number of occurrences of token
                alphas[kk][tt][MM - 1] = qzz[kk] * sum;

                //Computing alphas(:,:,0:_MM-2) and chishat
                for (int mm = 0; mm < MM - 1; mm++) {
                    sum = 0;
                    for (int ii = 0; ii < doc.ss.size(); ii++)
                        sum += qtt[ii][kk][tt] * contEloglik[ii][mm] * doc.ss.size(); // multiply number of occurences of token;
                    alphas[kk][tt][mm] = qzz[kk] * sum;
                    result.docChishat[kk][tt][mm] += Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                }
            }
        }


        //Update \rhoshat
        sum = qzz[KK - 1];
        for (int kk = KK - 2; kk >= 0; kk--) {
            result.docRhoshat[kk][0] = 1 + ngroups * qzz[kk];
            result.docRhoshat[kk][1] = gg + ngroups * sum;
            sum += qzz[kk];

        }

        //Update \zetashat
        for (int kk = 0; kk < KK; kk++) {
            // Computing the last TT
            double[] sumzetas = new double[TT];
            for (int ii = 0; ii < doc.ss.size(); ii++)
                for (int tt = 0; tt < TT; tt++) {
                    sumzetas[tt] += qtt[ii][kk][tt]; // multiply number of occurrences of token;;
                }
            sum = sumzetas[TT - 1];
            for (int tt = TT - 2; tt >= 0; tt--) {
                result.docZetashat[kk][tt][0] = 1 + ngroups * qzz[kk] * sumzetas[tt];
                result.docZetashat[kk][tt][1] = aa + ngroups * qzz[kk] * sum;
                sum += sumzetas[tt];

            }
        }

        //Update \varphishat
        double[] sumkt = new double[MM];

        for (int mm = 0; mm < MM; mm++) {
            for (int kk = 0; kk < KK; kk++) {
                for (int tt = 0; tt < TT; tt++) {
                    sumkt[mm] += qcc[kk][tt][mm];
                }
            }
        }
        sum = sumkt[MM - 1];
        for (int mm = MM - 2; mm >= 0; mm--) {

            result.docVarphishat[mm][0] = 1 + ngroups * sumkt[mm];
            result.docVarphishat[mm][1] = ee + ngroups * sum;
            sum += sumkt[mm];

        }

        // Update \lambdashat
        double[][] dataProb = new double[doc.ss.size()][MM];
        for (int ii = 0; ii < doc.ss.size(); ii++)
            for (int mm = 0; mm < MM; mm++) {
                for (int kk = 0; kk < KK; kk++)
                    for (int tt = 0; tt < TT; tt++) {
                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
                    }
                try {
                    result.docLambdahat.get(mm).add(doc.ss.get(ii), ngroups / batchCount * dataProb[ii][mm]); // multiply number of occurrences of token
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        //Updating \alphashat
        try {
            for (int kk = 0; kk < KK; kk++) {
                result.docAlphahat.get(kk).add(doc.xx, ngroups / batchCount * qzz[kk]);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        // This is new command to save qzz for computing clustering performance
        result.qzz = new HashMap();
        result.qzz.put(doc.id, qzz);
        // Computing running time
        innerTimer.stop();
        System.out.println("\t\tFinished document " + jj + " elapse=" + innerTimer.getElaspedSeconds());
        return result;
    }

    public static MC2MapReduceOutput reduceOutput(MC2MapReduceOutput result1, MC2MapReduceOutput result2) {


        MC2MapReduceOutput result = result1;
//        HashMap<Integer,double[]> temp=new HashMap<>(result1.qzz.size()+result2.qzz.size());
//        for (int ii=0;ii<result1.qzz.size();ii++)
//            temp.p
        result.qzz.putAll(result2.qzz);


        for (int kk = 0; kk < KK; kk++) {
            for (int tt = 0; tt < TT; tt++) {
                for (int mm = 0; mm < MM - 1; mm++) {
                    result.docChishat[kk][tt][mm] += result2.docChishat[kk][tt][mm];
                }
                if (tt < TT - 1) {
                    result.docZetashat[kk][tt][0] += result2.docZetashat[kk][tt][0];
                    result.docZetashat[kk][tt][1] += result2.docZetashat[kk][tt][1];
                }
            }
            if (kk < KK - 1) {
                result.docRhoshat[kk][0] += result2.docRhoshat[kk][0];
                result.docRhoshat[kk][1] += result2.docRhoshat[kk][1];
            }
            try {
                result.docAlphahat.get(kk).plus(result2.docAlphahat.get(kk));
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
        //Update \varphishat
        for (int mm = 0; mm < MM; mm++) {
            try {
                // Update \lambdashat
                result.docLambdahat.get(mm).plus(result2.docLambdahat.get(mm));

            } catch (Exception e) {
                e.printStackTrace();
            }
            if (mm < MM - 1) {
                result.docVarphishat[mm][0] += result2.docVarphishat[mm][0];
                result.docVarphishat[mm][1] += result2.docVarphishat[mm][1];
            }
        }


        return result;
    }
//    public static MC2MapReduceOutput reduceOutput(List lsOutput ){
//        MC2MapReduceOutput result = new MC2MapReduceOutput();
//        //Ouput variables
//        result.docRhoshat = new double[KK - 1][2];
//        result.docChishat = new double[KK][TT][MM - 1];
//        result.docZetashat = new double[KK][TT - 1][2];
//        result.docVarphishat = new double[MM - 1][2];
//        result.docLambdahat = new ArrayList<BayesianComponent>();
//        result.docAlphahat = new ArrayList<BayesianComponent>();
//
//
//        // Context
//        for (int kk = 0; kk < KK; kk++) {
//            result.docAlphahat.add((BayesianComponent) q0.clone());
//        }
//
//        //Content
//        for (int mm = 0; mm < MM; mm++) {
//            result.docLambdahat.add((BayesianComponent) p0.clone());
//        }
//        for(Object out:lsOutput) {
//            for (int kk = 0; kk < KK; kk++) {
//                for (int tt = 0; tt < TT; tt++) {
//                    for (int mm = 0; mm < MM - 1; mm++) {
//                        result.docChishat[kk][tt][mm] += ((MC2MapReduceOutput)out).docChishat[kk][tt][mm] ;
//                    }
//                    if (tt < TT - 1) {
//                        result.docZetashat[kk][tt][0] += ((MC2MapReduceOutput)out).docZetashat[kk][tt][0];
//                        result.docZetashat[kk][tt][1] += ((MC2MapReduceOutput)out).docZetashat[kk][tt][1] ;
//                    }
//                }
//                if (kk < KK - 1) {
//                    result.docRhoshat[kk][0] += ((MC2MapReduceOutput)out).docRhoshat[kk][0] ;
//                    result.docRhoshat[kk][1] += ((MC2MapReduceOutput)out).docRhoshat[kk][1] ;
//                }
//                try {
//                    result.docAlphahat.get(kk).plus(((MC2MapReduceOutput)out).docAlphahat.get(kk));
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//            //Update \varphishat
//            for (int mm = 0; mm < MM; mm++) {
//                try {
//                    result.docLambdahat.get(mm).plus(((MC2MapReduceOutput)out).docLambdahat.get(mm));
//
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//                if (mm < MM - 1) {
//                    result.docVarphishat[mm][0] += ((MC2MapReduceOutput)out).docVarphishat[mm][0] ;
//                    result.docVarphishat[mm][1] += ((MC2MapReduceOutput)out).docVarphishat[mm][1];
//                }
//            }
//        }
//        return result;
//    }

    public static HashMap<Object, Integer> GetUniqueValue(ArrayList<Object> values) {
        HashMap<Object, Integer> uniqueValues = new HashMap<Object, Integer>();

        for (int ii = 0; ii < values.size(); ii++) {
            Integer val = uniqueValues.get(values.get(ii));
            if (val == null)
                uniqueValues.put(values.get(ii), 1);
            else
                uniqueValues.put(values.get(ii), val + 1);
        }
        return uniqueValues;

    }

    public static void smallDataExp() {
        int V = 10;
        int VContext = 2;
        int ngroups = 3;
        int[] numData = new int[ngroups];
        numData[0] = 4;
        numData[1] = 5;
        numData[2] = 4;

        int[][] ss = new int[3][];

        ss[0] = new int[4];
        ss[0][0] = 1;
        ss[0][1] = 1;
        ss[0][2] = 4;
        ss[0][3] = 3;

        ss[1] = new int[5];
        ss[1][0] = 0;
        ss[1][1] = 1;
        ss[1][2] = 2;
        ss[1][3] = 9;
        ss[1][4] = 3;

        ss[2] = new int[4];
        ss[2][0] = 9;
        ss[2][1] = 4;
        ss[2][2] = 5;
        ss[2][3] = 6;

        int[] xx = new int[ngroups];//time
        xx[0] = 0;
        xx[1] = 0;
        xx[2] = 1;

        // base measure of topic
        double sym = 0.01;
        int trunM = 4;
        MultinomialDirichlet H = new MultinomialDirichlet(V, sym * V);

        // base measure of author
        double symAuthor = 0.01;
        int trunK = 2;
        int trunT = 3;
        MultinomialDirichlet L = new MultinomialDirichlet(VContext, symAuthor * VContext);

        double aa = 1;
        double ee = 20;
        double vv = 1;
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ").setMaster("local");

        JavaSparkContext jsc = new JavaSparkContext(conf);
        MC2StochasticVariationalInferenceSpark mc2 = new MC2StochasticVariationalInferenceSpark(trunK, trunT, trunM, ee, aa, vv, L, H, jsc);
        mc2.loadData(xx, ss);
        mc2.initialize();
        String strOutMat = "D:\\";
//        mc2.sviOptimizer(2, 1, 1, 0.6, strOutMat);
        mc2.sviCategoryOptimizer(2, 2, 1, 0.6, strOutMat);

        //mc2.SVIParallelOptimizer(1, 2, 1, .6, strOutMat);
        //wsndp.CollapsedGibbsInference(initK, initM, 10, 1000, 1, stirlingFilename, strOutMat, cp, smooth, stirlingFolder);
        //System.Console.WriteLine(Environment.CurrentDirectory.ToString());


    }

    public static void NIPSExperiment_AuthorContext() {


        //string folder = @"P:\WithVu_Experiment\csdp shared\csdp shared NIPS dataset\splitedmatfiles_wsndp\";
        String strMat = "F:\\MyPhD\\Code\\CSharp\\MC2_VU_CODE\\csdp shared\\csdp shared NIPS dataset\\nips_wsndp_Csharp_author_90pctrain.mat";

        MC2InputDataMultCat data = MatlabJavaConverter.readMC2DataFromMatFiles("V", "VAuthor", strMat);

        System.out.println("Vocabulary Size = " + data.contentDim);

        System.out.println("Author Vocabulary Size = " + data.contextDim);

        System.out.println("Number of Documents = " + data.ngroups);


        // base measure of topic
        double sym = 0.01;
        int trunM = 100;
        MultinomialDirichlet H = new MultinomialDirichlet(data.contentDim, sym * data.contentDim);

        // base measure of author
        double symAuthor = 0.01;
        int trunK = 20;
        int trunT = 50;
        MultinomialDirichlet L = new MultinomialDirichlet(data.contextDim, symAuthor * data.contextDim);


        double aa = 10;
        double ee = 10;
        double vv = 10;

        System.out.println("Creating SVI inference engine...");
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ");

        JavaSparkContext jsc = new JavaSparkContext(conf);
        MC2StochasticVariationalInferenceSpark mc2 = new MC2StochasticVariationalInferenceSpark(trunK, trunT, trunM, ee, aa, vv, L, H, jsc);

        mc2.loadData(data.xx, data.ss);

        System.out.println("Initializing ...");
        mc2.initialize();
        // string strOutMat = @"D:\Nips_Author_SVI_CSharp_Parallel\";
        String strOutMat = "D:\\Nips_Author_SVI_CSharp_SPARK\\";

        // SVI params
        int numIter = 2;
        int batchSize = 50;
        double varrho = 1;
        double iota = 0.6;

        System.out.println("Running ...");
        mc2.sviCategoryOptimizer(numIter, batchSize, varrho, iota, strOutMat);
    }

    public static void NIPSExperiment_AuthorContext_DataFromFile(String propFileName) {
        Properties prop = new Properties();
        propFileName = "config.properties";
        InputStream input = null;
        try {
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
        double contentSym = Double.valueOf(prop.getProperty("mc2.contentDirichletSym"));
        double contextSym = Double.valueOf(prop.getProperty("mc2.contextDirichletSym"));
        int trunM = Integer.valueOf(prop.getProperty("mc2.trunM"));
        int trunK = Integer.valueOf(prop.getProperty("mc2.trunK"));
        int trunT = Integer.valueOf(prop.getProperty("mc2.trunT"));
        double aa = Double.valueOf(prop.getProperty("mc2.aa"));
        double ee = Double.valueOf(prop.getProperty("mc2.ee"));
        double vv = Double.valueOf(prop.getProperty("mc2.vv"));
        String strContent = prop.getProperty("mc2.contentPath");
        String strContext = prop.getProperty("mc2.contextPath");
        String strOutMat = prop.getProperty("mc2.outFolderPath");
        String contextType = prop.getProperty("mc2.contextType");

        //MC2InputDataMultCat data= MatlabJavaConverter.readMC2DataFromMatFiles("V","VAuthor", strMat);
        MC2InputDataMultCat data = new MC2InputDataMultCat();
        data.contentDim = 13649;
        data.contextDim = 2037;


        System.out.println("Vocabulary Size = " + data.contentDim);

        System.out.println("Author Vocabulary Size = " + data.contextDim);

        // base measure of topic
        MultinomialDirichlet H = new MultinomialDirichlet(data.contentDim, contentSym * data.contentDim);

        // base measure of author
        MultinomialDirichlet L = new MultinomialDirichlet(data.contextDim, contextSym * data.contextDim);

        System.out.println("Creating SVI inference engine...");
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ");

        JavaSparkContext jsc = new JavaSparkContext(conf);

        MC2StochasticVariationalInferenceSpark mc2 = new MC2StochasticVariationalInferenceSpark(trunK, trunT, trunM, ee, aa, vv, L, H, jsc);
        try {
            mc2.loadData(strContent, strContext, true, contextType);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Initializing ...");
        mc2.initialize();
        // string strOutMat = @"D:\Nips_Author_SVI_CSharp_Parallel\";

        // SVI params
        int numIter = 2;
        int batchSize = 50;
        double varrho = 1;
        double iota = 0.6;

        System.out.println("Running ...");
        mc2.sviCategoryOptimizer(numIter, batchSize, varrho, iota, strOutMat);

    }

    public static void NUSWIDEExperiment_TagsContext_DataFromFile(String propFileName) {
        Properties prop = new Properties();
        propFileName = "config.properties";
        InputStream input = null;
        try {
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
        double contentSym = Double.valueOf(prop.getProperty("mc2.contentDirichletSym"));
        double contextSym = Double.valueOf(prop.getProperty("mc2.contextDirichletSym"));
        int trunM = Integer.valueOf(prop.getProperty("mc2.trunM"));
        int trunK = Integer.valueOf(prop.getProperty("mc2.trunK"));
        int trunT = Integer.valueOf(prop.getProperty("mc2.trunT"));
        double aa = Double.valueOf(prop.getProperty("mc2.aa"));
        double ee = Double.valueOf(prop.getProperty("mc2.ee"));
        double vv = Double.valueOf(prop.getProperty("mc2.vv"));
        String strContent = prop.getProperty("mc2.contentPath");
        String strContext = prop.getProperty("mc2.contextPath");
        String strOutMat = prop.getProperty("mc2.outFolderPath");
        String contextType = prop.getProperty("mc2.contextType");

        //MC2InputDataMultCat data= MatlabJavaConverter.readMC2DataFromMatFiles("V","VAuthor", strMat);
        MC2InputDataMultCat data = new MC2InputDataMultCat();
        data.contentDim = 13649;
        data.contextDim = 2037;


        System.out.println("Vocabulary Size = " + data.contentDim);

        System.out.println("Author Vocabulary Size = " + data.contextDim);

        // base measure of topic
        MultinomialDirichlet H = new MultinomialDirichlet(data.contentDim, contentSym * data.contentDim);

        // base measure of author
        MultinomialDirichlet L = new MultinomialDirichlet(data.contextDim, contextSym * data.contextDim);

        System.out.println("Creating SVI inference engine...");
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ");

        JavaSparkContext jsc = new JavaSparkContext(conf);

        MC2StochasticVariationalInferenceSpark mc2 = new MC2StochasticVariationalInferenceSpark(trunK, trunT, trunM, ee, aa, vv, L, H, jsc);

        try {
            mc2.loadData(strContent, strContext, true, contextType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Initializing ...");
        mc2.initialize();
        // string strOutMat = @"D:\Nips_Author_SVI_CSharp_Parallel\";

        // SVI params
        int numIter = 2;
        int batchSize = 50;
        double varrho = 1;
        double iota = 0.6;

        System.out.println("Running ...");
        mc2.sviCategoryOptimizer(numIter, batchSize, varrho, iota, strOutMat);

    }

    public static void computeTestClustering(String testFilePath) {

    }

}
