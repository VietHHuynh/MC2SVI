package org.bnpstat.mc2;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.bnpstat.maths.SparseVector;
import org.bnpstat.stats.conjugacy.BayesianComponent;
import org.bnpstat.stats.conjugacy.MultinomialDirichlet;
import org.bnpstat.util.MatlabJavaConverter;
import org.bnpstat.util.PerformanceTimer;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by hvhuynh on 11/9/2015.
 */
public class MC2StochasticVariationalInference {
    private BayesianComponent q0;
    private double aa;

    private BayesianComponent p0;
    private double ee;
    private double gg;


    private int ngroups;
    private int[] numdata;
    private ArrayList<Object> xx; //content data
    private ArrayList<Object>[] ss; //context data
    private int KK;
    private int TT;
    private int MM;

    // Model variables
    private ArrayList<BayesianComponent> qq;
    private ArrayList<BayesianComponent> pp;
    private double[] qzz;
    private double[][][] qcc;
    private double[][] rhos;
    private double[][][] zetas;
    private double[][] varphis;

    /**
     * Constructor
     */

    public MC2StochasticVariationalInference(int KKTrun, int TTTrun, int MMTrun, double epsison, double alpha, double beta, BayesianComponent q0Context, BayesianComponent p0Content) {
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


    }

    /**
     * Initialize parameters and hyperparameters for the VB learning
     */
    public void initialize() {
        Random rnd = new Random();
        rnd.setSeed(6789);


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
                    gt.mapAddToSelf(gt.getL1Norm());
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
        double sumrhos = rnd.nextInt(100); // last KK
        for (int kk = KK - 2; kk >= 0; kk--) {
            rhos[kk][0] = 1 + rnd.nextInt(100);
            rhos[kk][1] = gg + sumrhos;
            sumrhos += rhos[kk][0] - 1;
        }

        //zetas
        zetas = new double[KK][TT - 1][2];
        for (int kk = 0; kk < KK; kk++) {
            double sumzetas = rnd.nextInt(100); // last TT
            for (int tt = TT - 2; tt >= 0; tt--) {
                zetas[kk][tt][0] = 1 + rnd.nextInt(100);
                zetas[kk][tt][1] = aa + sumzetas;
                sumzetas += zetas[kk][tt][0] - 1;
            }
        }

        //varphis
        varphis = new double[MM - 1][2];
        double sumvarphis = rnd.nextInt(100); // last KK
        for (int mm = MM - 2; mm >= 0; mm--) {
            varphis[mm][0] = 1 + rnd.nextInt(100);
            varphis[mm][1] = ee + sumvarphis;
            sumvarphis += varphis[mm][0] - 1;
        }

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

    /**
     * Load data into local variables xx, and ss
     *
     * @param inXX - data loaded into xx: a list of Object
     * @param inSS - data loaded into ss: a two-dimension list of double
     */
    public void loadData(Object[] inXX, double[][] inSS) {
        ngroups = inXX.length;
        numdata = new int[ngroups];

        xx = new ArrayList<Object>();
        ss = new ArrayList[ngroups];

        for (int jj = 0; jj < ngroups; jj++) {
            xx.add(inXX[jj]);
            ss[jj] = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                ss[jj].add(inSS[jj][ii]);
            }
        }
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

        xx = new ArrayList<Object>();
        ss = new ArrayList[ngroups];

        for (int jj = 0; jj < ngroups; jj++) {
            xx.add(inXX[jj]);
            ss[jj] = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                ss[jj].add(inSS[jj][ii]);
            }
        }
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

        xx = new ArrayList<Object>();
        ss = new ArrayList[ngroups];

        for (int jj = 0; jj < ngroups; jj++) {
            xx.add(inXX[jj]);
            ss[jj] = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                ss[jj].add(inSS[jj][ii]);
            }
        }
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

        xx = new ArrayList<Object>();
        ss = new ArrayList[ngroups];

        for (int jj = 0; jj < ngroups; jj++) {
            xx.add(inXX[jj]);
            ss[jj] = new ArrayList<Object>();
            numdata[jj] = inSS[jj].length;
            for (int ii = 0; ii < numdata[jj]; ii++) {
                ss[jj].add(inSS[jj][ii]);
            }
        }
    }

    /**
     * Computing stick breaking expectation with DP truncation
     * @param hyperprams - hyper parameters of beta distributions for sticks
     * @return
     */
    private double[] computeStickBreakingExpectation(double[][] hyperprams) throws  Exception
    {
        if (hyperprams[0].length!= 2)
            throw new Exception("The dimension of hyperparameters is not correct");
        int KK = hyperprams.length + 1;
        double[] EStick = new double[KK];

        double temp = Gamma.digamma(hyperprams[0][0] + hyperprams[0][1]);
        EStick[0] = Gamma.digamma(hyperprams[0][0]) - temp;// The first stick

        double sum = Gamma.digamma(hyperprams[0][1]) - temp;
        for (int kk = 1; kk < KK - 1; kk++)
        {
            temp = Gamma.digamma(hyperprams[kk][0] + hyperprams[kk][1]);
            EStick[kk] =Gamma.digamma(hyperprams[kk][0]) - temp + sum;
            sum += Gamma.digamma(hyperprams[kk][1]) - temp;
        }
        EStick[KK - 1] = sum;// The last stick
        return  EStick;
    }

    /**
     * Optimizing routing using SVI
     * @param numIter - number of total iteration, each iteration will visit all data points. This is also usually called epoch.
     * @param batchSize - mini-batch size of documents used in each update
     * @param varrho - hyperparameter for computing learning rate (1+iota)^(-varrho)
     * @param iota - hyperparameter for computing learning rate (1+iota)^(-varrho)
     * @param strOutFolder - output folder to export result after each update (= mini-batch)
     */
    public void sviOptimizer(int numIter, int batchSize, double varrho, double iota, String strOutFolder)
    {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++)
        {
            for (int tt = 0; tt < TT; tt++)
            {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++)
        {
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
        for (int iter = 1; iter <= numIter; iter++)
        {
            System.out.println("Iteration "+iter+" K=" +KK+ " M="+MM+" elapse="+ elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++)
            {
                timer.start();

                // Imtermediate updates for global variable  for mini-batch
                double[][] rhoshat = new double[KK - 1][2];
                double[][][] chishat = new double[KK][TT][MM - 1];
                double[][][] zetashat = new double[KK][TT - 1][2];
                double[][] varphishat = new double[MM - 1][2];

                ArrayList<BayesianComponent> alphahat = new ArrayList<BayesianComponent>();
                ArrayList<BayesianComponent> lambdahat = new ArrayList<BayesianComponent>();

                // Context atoms
                for (int kk = 0; kk < KK; kk++)
                {
                    alphahat.add((BayesianComponent)q0.clone());
                }

                //Content atoms
                for (int mm = 0; mm < MM; mm++)
                {
                    lambdahat.add((BayesianComponent) p0.clone());
                }

                System.out.println("\t Running mini-batch "+ba+" from document # "+(noProcessedDocs+1)+" to "+(noProcessedDocs + batchCount[ba])+", elapse= "+ elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick=null;
                double[][] Elowerstick=null;
                double[] Esharestick=null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick= new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                }catch (Exception e){
                    e.printStackTrace();
                }



                // Start to compute each group in the current batch
                for (int jj = noProcessedDocs; jj < noProcessedDocs + batchCount[ba]; jj++)
                {
                    innerTimer.start();

                    System.out.println("\t Running document "+ (jj - noProcessedDocs + 1)+" out of "+batchCount[ba]+", elapse= "+innerElapse);

                    // Computing expectation of log likelihood
                    double[][] contEloglik = new double[numdata[jj]][MM];
                    double[] cxtEloglik = new double[KK];
                    try {
                        for (int kk = 0; kk < KK; kk++) {
                            cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(xx.get(jj));
                        }
                        for (int ii = 0; ii < numdata[jj]; ii++) {
                            for (int mm = 0; mm < MM; mm++) {
                                contEloglik[ii][mm]=pp.get(mm).expectationLogLikelihood(ss[jj].get(ii));
                            }
                        }
                    }catch (Exception e){
                        e.printStackTrace();
                    }

                    //Update local parameters \kappa,\vartheta

                    //Computing \kappa

                    double[][][] qtt = new double[numdata[jj]][KK][TT];
                    double[][][] unormalizedQtt = new double[numdata[jj]][KK][TT]; // used for computing qzz
                    double[][] maxQtt = new double[numdata[jj]][KK];
                    for (int ii = 0; ii < numdata[jj]; ii++)
                    {
                        for (int kk = 0; kk < KK; kk++)
                        {
                            maxQtt[ii][kk] = Double.NEGATIVE_INFINITY;

                            for (int tt = 0; tt < TT; tt++)
                            {
                                unormalizedQtt[ii][kk][tt] = 0;
                                for (int mm = 0; mm < MM; mm++)
                                {
                                    unormalizedQtt[ii][kk][tt] += qcc[kk][tt][mm] * contEloglik[ii][mm];
                                }
                                unormalizedQtt[ii][kk][tt] += Elowerstick[kk][tt];
                                if (unormalizedQtt[ii][kk][tt] > maxQtt[ii][kk])
                                    maxQtt[ii][kk] = unormalizedQtt[ii][kk][tt];
                            }
                            // Convert to exponential
                            double sum = 0;
                            for (int tt = 0; tt < TT; tt++)
                            {
                                unormalizedQtt[ii][kk][tt] = FastMath.exp(unormalizedQtt[ii][kk][tt] - maxQtt[ii][kk]);
                                sum += unormalizedQtt[ii][kk][tt];
                            }
                            //Normalize
                            for (int tt = 0; tt < TT; tt++)
                                qtt[ii][kk][tt] = unormalizedQtt[ii][kk][tt]/ sum;
                        }
                    }

                    //Computing \vartheta
                    double[] qzz = new double[KK];
                    double maxQzz = Double.NEGATIVE_INFINITY;
                    for (int kk = 0; kk < KK; kk++)
                    {
                        qzz[kk] = 0;
                        for (int ii = 0; ii < numdata[jj]; ii++)
                        {
                            double sum = 0;
                            for (int tt = 0; tt < TT; tt++)
                                sum += unormalizedQtt[ii][kk][tt];
                            qzz[kk] += maxQtt[ii][kk] + FastMath.log(sum);
                        }
                        if (qzz[kk] > maxQzz)
                            maxQzz = qzz[kk];
                        qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
                    }

                    // Convert to exponential
                    double sum = 0;
                    for (int kk = 0; kk < KK; kk++)
                    {
                        qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
                        sum += qzz[kk];
                    }
                    //Normalize
                    for (int kk = 0; kk < KK; kk++)
                    {
                        qzz[kk] = qzz[kk] / sum;
                    }

                    //Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat

                    //Update \chihat
                    double[][][] alphas = new double[KK][TT][MM];

                    //Computing alphas(:,:,_MM-1)
                    for (int kk = 0; kk < KK; kk++)
                    {
                        for (int tt = 0; tt < TT; tt++)
                        {
                            sum = 0;
                            for (int ii = 0; ii < numdata[jj]; ii++)
                                sum += qtt[ii][kk][tt] * contEloglik[ii][MM - 1];
                            alphas[kk][tt][MM - 1] = qzz[kk] * sum;
                        }
                    }

                    //Computing alphas(:,:,0:_MM-2) and chishat
                    for (int kk = 0; kk < KK; kk++)
                    {
                        for (int tt = 0; tt < TT; tt++)
                        {
                            for (int mm = 0; mm < MM - 1; mm++)
                            {
                                sum = 0;
                                for (int ii = 0; ii < numdata[jj]; ii++)
                                    sum += qtt[ii][kk][tt] * contEloglik[ii][mm];
                                alphas[kk][tt][mm] = qzz[kk] * sum;
                                chishat[kk][tt][mm] += Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                            }
                        }
                    }

                    //Update \rhoshat
                    sum = qzz[KK - 1];
                    for (int kk = KK - 2; kk >= 0; kk--)
                    {
                        rhoshat[kk][0] += 1 + ngroups * qzz[kk];
                        rhoshat[kk][1] += gg + ngroups * sum;
                        sum += qzz[kk];
                    }

                    //Update \zetashat
                    for (int kk = 0; kk < KK; kk++)
                    {
                        // Computing the last TT
                        double[] sumzetas = new double[TT];
                        for (int ii = 0; ii < numdata[jj]; ii++)
                            for (int tt = 0; tt < TT; tt++)
                            {
                                sumzetas[tt] += qtt[ii][kk][tt];
                            }
                        sum = sumzetas[TT - 1];
                        for (int tt = TT - 2; tt >= 0; tt--)
                        {
                            zetashat[kk][tt][0] += 1 + ngroups * qzz[kk] * sumzetas[tt];
                            zetashat[kk][tt][1] += aa + ngroups * qzz[kk] * sum;
                            sum += sumzetas[tt];
                        }
                    }

                    //Update \varphishat
                    double[] sumkt = new double[MM];

                    for (int mm = 0; mm < MM; mm++)
                    {
                        for (int kk = 0; kk < KK; kk++)
                        {
                            for (int tt = 0; tt < TT; tt++)
                            {
                                sumkt[mm] += qcc[kk][tt][mm];
                            }
                        }
                    }
                    sum = sumkt[MM - 1];
                    for (int mm = MM - 2; mm >= 0; mm--)
                    {
                        varphishat[mm][0] += 1 + ngroups * sumkt[mm];
                        varphishat[mm][1] += ee + ngroups * sum;
                        sum += sumkt[mm];
                    }

                    // Update \lambdashat
                    try {
                        double[][] dataProb = new double[numdata[jj]][MM];
                        for (int ii = 0; ii < numdata[jj]; ii++)
                            for (int mm = 0; mm < MM; mm++) {
                                for (int kk = 0; kk < KK; kk++)
                                    for (int tt = 0; tt < TT; tt++) {
                                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
                                    }
                                lambdahat.get(mm).add(ss[jj].get(ii), ngroups / batchCount[ba] * dataProb[ii][mm]);
                            }

                        //Updating \alphashat
                        for (int kk = 0; kk < KK; kk++) {
                            alphahat.get(kk).add(xx.get(jj), ngroups / batchCount[ba] * qzz[kk]);
                        }
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                    // Computing running time
                    innerTimer.stop();
                    innerElapse = innerTimer.getElaspedSeconds();

                }

                // Update global parameters
                double varpi = FastMath.pow(ba + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++)
                {
                    for (int tt = 0; tt < TT; tt++)
                    {
                        for (int mm = 0; mm < MM - 1; mm++)
                        {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++)
                        {
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
                for (int kk = 0; kk < KK - 1; kk++)
                {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * rhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * rhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++)
                {
                    for (int tt = 0; tt < TT - 1; tt++)
                    {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * zetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * zetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++)
                {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * varphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * varphishat[mm][1];
                }
                try{
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++)
                    {
                        qq.get(kk).stochasticUpdate(alphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++)
                    {
                        pp.get(mm).stochasticUpdate(lambdahat.get(mm), varpi);
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder+ "batch_" + ba + "iter" + iter + "_mc2_svi.mat";
                System.out.println("\tExporting SVI output to Mat files...");
                try {
                    MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);
                }catch (Exception e){
                    e.printStackTrace();
                }

            }
        }
    }
    public void sviCategoryOptimizer(int numIter, int batchSize, double varrho, double iota, String strOutFolder)
    {
        double elapse = 0;
        double innerElapse = 0;

        PerformanceTimer timer = new PerformanceTimer();
        PerformanceTimer innerTimer = new PerformanceTimer();

        // Convert qcc to natual paramters
        double[][][] chis = new double[KK][TT][MM - 1];
        for (int kk = 0; kk < KK; kk++)
        {
            for (int tt = 0; tt < TT; tt++)
            {
                for (int mm = 0; mm < MM - 1; mm++)
                    chis[kk][tt][mm] = FastMath.log(qcc[kk][tt][mm]) - FastMath.log(qcc[kk][tt][MM - 1]);
            }
        }

        // Computing number of document for each mini-batch
        int nBatches = ngroups / batchSize;

        if ((ngroups - nBatches * batchSize) > 0)
            nBatches = nBatches + 1;
        int[] batchCount = new int[nBatches];
        for (int i = 0; i < nBatches - 1; i++)
        {
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
        for (int iter = 1; iter <= numIter; iter++)
        {
            System.out.println("Iteration "+iter+" K=" +KK+ " M="+MM+" elapse="+ elapse);
            int noProcessedDocs = 0;
            for (int ba = 0; ba < nBatches; ba++)
            {
                timer.start();

                // Imtermediate updates for global variable  for mini-batch
                double[][] rhoshat = new double[KK - 1][2];
                double[][][] chishat = new double[KK][TT][MM - 1];
                double[][][] zetashat = new double[KK][TT - 1][2];
                double[][] varphishat = new double[MM - 1][2];

                ArrayList<BayesianComponent> alphahat = new ArrayList<BayesianComponent>();
                ArrayList<BayesianComponent> lambdahat = new ArrayList<BayesianComponent>();

                // Context
                for (int kk = 0; kk < KK; kk++)
                {
                    alphahat.add((BayesianComponent)q0.clone());
                }

                //Content
                for (int mm = 0; mm < MM; mm++)
                {
                    lambdahat.add((BayesianComponent) p0.clone());
                }

                System.out.println("\t Running mini-batch "+ba+" from document # "+(noProcessedDocs+1)+" to "+(noProcessedDocs + batchCount[ba])+", elapse= "+ elapse);

                //Computing expectation of stick-breakings Etopstick, Elowerstick, Esharestick

                double[] EtopStick=null;
                double[][] Elowerstick=null;
                double[] Esharestick=null;
                try {

                    //Computing E[ln\beta_k] - we call the expection of top stick - Etopstick
                    EtopStick = computeStickBreakingExpectation(rhos);

                    //Computing E[ln\pi_kl] - we call the expection of lower stick - Elowerstick
                    Elowerstick= new double[KK][TT];
                    for (int kk = 0; kk < KK; kk++) {
                        Elowerstick[kk] = computeStickBreakingExpectation(zetas[kk]);
                    }
                    //Computing E[ln\epsilon_m] - we call the expection of sharing stick - Esharestick
                    Esharestick = computeStickBreakingExpectation(varphis);

                }catch (Exception e){
                    e.printStackTrace();
                }

                HashMap<Integer, double[]> cachedContEloglik = new HashMap<Integer, double[]>();

                PerformanceTimer myTimer = new PerformanceTimer();
                //double total = 0;
                myTimer.start();

                for (int jj = noProcessedDocs; jj < noProcessedDocs + batchCount[ba]; jj++)
                {
                    for (int ii = 0; ii < numdata[jj]; ii++)
                    {
                        if (!cachedContEloglik.containsKey((Integer) ss[jj].get(ii)))
                        {
                            cachedContEloglik.put((Integer) ss[jj].get(ii), null);
                        }
                    }
                }
                myTimer.stop();
                System.out.println("\tRunning time for  getting word id: " + myTimer.getElaspedSeconds());
                myTimer.start();
                ArrayList<Integer> keys = new ArrayList<Integer>(cachedContEloglik.keySet());
                double[][] tempEll = MultinomialDirichlet.expectationLogLikelihood(keys, pp);
                for (int ii = 0; ii < keys.size(); ii++)
                    cachedContEloglik.put(keys.get(ii), tempEll[ii]);
                myTimer.stop();
                System.out.println("\tRunning time for  computing expectation: " + myTimer.getElaspedSeconds());

                MC2MapReduceOutput[] taskOutputs = new MC2MapReduceOutput[batchCount[ba]];

                // Context
                for (int kk = 0; kk < KK; kk++)
                {
                    alphahat.add((BayesianComponent) q0.clone());
                }

                //Content
                for (int mm = 0; mm < MM; mm++)
                {
                    lambdahat.add((BayesianComponent) p0.clone());
                }


                for (int i = 0; i < batchCount[ba]; ++i)
                {
                    int docID =i + noProcessedDocs;
                    taskOutputs[i] = computeParamEachDoc(docID, batchCount[ba], cachedContEloglik, EtopStick, Elowerstick, Esharestick);
                }

                // Aggregate data
                for (int ii = 0; ii < batchCount[ba]; ii++)
                {
                    for (int kk = 0; kk < KK; kk++)
                    {
                        for (int tt = 0; tt < TT; tt++)
                        {
                            for (int mm = 0; mm < MM-1; mm++)
                            {
                                chishat[kk][tt][mm] += taskOutputs[ii].docChishat[kk][tt][mm];
                            }
                        }
                    }

                    for (int kk = 0; kk < KK - 1; kk++)
                    {
                        rhoshat[kk][0] += taskOutputs[ii].docRhoshat[kk][0];
                        rhoshat[kk][1] += taskOutputs[ii].docRhoshat[kk][1];
                    }

                    //Update \zetashat
                    for (int kk = 0; kk < KK; kk++)
                    {
                        for (int tt = 0; tt < TT - 1; tt++)
                        {
                            zetashat[kk][tt][0] += taskOutputs[ii].docZetashat[kk][tt][0];
                            zetashat[kk][tt][1] += taskOutputs[ii].docZetashat[kk][tt][1];
                        }
                        //Updating \alphashat
                        try {
                            alphahat.get(kk).plus(taskOutputs[ii].docAlphahat.get(kk));
                        }catch (Exception e){
                            e.printStackTrace();
                        }
                    }
                    try {
                        //Update \varphishat
                        for (int mm = 0; mm < MM - 1; mm++) {
                            varphishat[mm][0] += taskOutputs[ii].docVarphishat[mm][0];
                            varphishat[mm][1] += taskOutputs[ii].docVarphishat[mm][1];

                            // Update \lambdashat
                            lambdahat.get(mm).plus(taskOutputs[ii].docLambdahat.get(mm));
                        }

                        lambdahat.get(MM - 1).plus(taskOutputs[ii].docLambdahat.get(MM - 1));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }

                // Update global parameters
                double varpi = FastMath.pow(ba + varrho, -iota);

                // Update \chis and qcc
                double[][] maxChis = new double[KK][TT];
                for (int kk = 0; kk < KK; kk++)
                {
                    for (int tt = 0; tt < TT; tt++)
                    {
                        for (int mm = 0; mm < MM - 1; mm++)
                        {
                            chis[kk][tt][mm] = (1 - varpi) * chis[kk][tt][mm] + varpi / batchCount[ba] * chishat[kk][tt][mm];
                            if (chis[kk][tt][mm] > maxChis[kk][tt]) maxChis[kk][tt] = chis[kk][tt][mm];
                        }
                        double sum = 0;
                        //Convert to exponential
                        for (int mm = 0; mm < MM - 1; mm++)
                        {
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
                for (int kk = 0; kk < KK - 1; kk++)
                {
                    rhos[kk][0] = (1 - varpi) * rhos[kk][0] + varpi / batchCount[ba] * rhoshat[kk][0];
                    rhos[kk][1] = (1 - varpi) * rhos[kk][1] + varpi / batchCount[ba] * rhoshat[kk][1];
                }

                //Update \tau_kt
                for (int kk = 0; kk < KK; kk++)
                {
                    for (int tt = 0; tt < TT - 1; tt++)
                    {
                        zetas[kk][tt][0] = (1 - varpi) * zetas[kk][tt][0] + varpi / batchCount[ba] * zetashat[kk][tt][0];
                        zetas[kk][tt][1] = (1 - varpi) * zetas[kk][tt][1] + varpi / batchCount[ba] * zetashat[kk][tt][1];
                    }
                }

                // Update \varphis
                for (int mm = 0; mm < MM - 1; mm++)
                {
                    varphis[mm][0] = (1 - varpi) * varphis[mm][0] + varpi / batchCount[ba] * varphishat[mm][0];
                    varphis[mm][1] = (1 - varpi) * varphis[mm][1] + varpi / batchCount[ba] * varphishat[mm][1];
                }
                try{
                    // Update \alpha
                    for (int kk = 0; kk < KK; kk++)
                    {
                        qq.get(kk).stochasticUpdate(alphahat.get(kk), varpi);
                    }

                    //Update \lambada
                    for (int mm = 0; mm < MM; mm++)
                    {
                        pp.get(mm).stochasticUpdate(lambdahat.get(mm), varpi);
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }

                // Computing running time
                timer.stop();
                elapse = timer.getElaspedSeconds();

                noProcessedDocs += batchCount[ba];

                //#region Export result to Matlab file
                String strOut = strOutFolder+ "batch_" + ba + "iter" + iter + "_mc2_svi.mat";
                System.out.println("Exporting SVI output to Mat files...");
                try {
                    MatlabJavaConverter.exportMC2SVIResultToMat(KK, TT, MM, qcc, rhos, zetas, varphis, ngroups, numdata, elapse, qq, pp, strOut, 1);
                }catch (Exception e){
                    e.printStackTrace();
                }

            }
        }
    }
    public MC2MapReduceOutput computeParamEachDoc(int docID, int batchCount, HashMap<Integer, double[]> cachedContEloglik,
                                               double[] EtopStick, double[][] Elowerstick, double[] Esharestick)
    {

        MC2MapReduceOutput result = new MC2MapReduceOutput();

        PerformanceTimer innerTimer = new PerformanceTimer();

        //region Ouput variables
        result.docRhoshat = new double[KK - 1][ 2];
        result.docChishat = new double[KK][TT][MM - 1];
        result.docZetashat = new double[KK][TT - 1][2];
        result.docVarphishat = new double[MM - 1][2];
        result.docLambdahat = new ArrayList<BayesianComponent>();
        result.docAlphahat = new ArrayList<BayesianComponent>();

        double sum = 0;

        // Context
        for (int kk = 0; kk < KK; kk++)
        {
            result.docAlphahat.add((BayesianComponent) q0.clone());
        }

        //Content
        for (int mm = 0; mm < MM; mm++)
        {
            result.docLambdahat.add((BayesianComponent) p0.clone());
        }

        int jj = docID;
        innerTimer.start();
        System.out.println("\t\tRunning document # " + jj);

        // Computing expectation of log likelihood
        HashMap<Object, Integer> contMap = GetUniqueValue(ss[jj]);
        ArrayList<Object> contData = new ArrayList<Object>(contMap.keySet());

        double[][] contEloglik = new double[contData.size()][MM];
        double[] cxtEloglik = new double[KK];
        try {
            for (int kk = 0; kk < KK; kk++) {
                cxtEloglik[kk] = qq.get(kk).expectationLogLikelihood(xx.get(jj));
            }
        }catch (Exception e){
            e.printStackTrace();
        }

        for (int ii = 0; ii < contData.size(); ii++)
        {
            double[] vals = cachedContEloglik.get(contData.get(ii));
            for (int mm = 0; mm < MM; mm++)
            {
                contEloglik[ii][mm] = vals[mm];
            }
        }

        //region Update local parameters \kappa,\vartheta

        //Computing \kappa

        double[][][] qtt = new double[contData.size()][KK][TT];
        double[][][] unormalizedQtt = new double[contData.size()][KK][TT]; // used for computing qzz
        double[][] maxQtt = new double[contData.size()][KK];
        for (int ii = 0; ii < contData.size(); ii++)
        {
            for (int kk = 0; kk < KK; kk++)
            {
                maxQtt[ii][kk] = Double.NEGATIVE_INFINITY;

                for (int tt = 0; tt < TT; tt++)
                {
                    unormalizedQtt[ii][kk][tt] = 0;
                    for (int mm = 0; mm < MM; mm++)
                    {
                        unormalizedQtt[ii][kk][tt] += qcc[kk][tt][mm] * contEloglik[ii][mm];
                    }
                    unormalizedQtt[ii][kk][tt] += Elowerstick[kk][tt];
                    if (unormalizedQtt[ii][kk][tt] > maxQtt[ii][kk])
                    maxQtt[ii][kk] = unormalizedQtt[ii][kk][tt];
                }
                // Convert to exponential
                sum = 0;
                for (int tt = 0; tt < TT; tt++)
                {
                    unormalizedQtt[ii][kk][tt] = FastMath.exp(unormalizedQtt[ii][kk][tt] - maxQtt[ii][kk]);
                    sum += unormalizedQtt[ii][kk][tt];
                }
                //Normalize
                for (int tt = 0; tt < TT; tt++)
                    qtt[ii][kk][tt] = unormalizedQtt[ii][kk][tt] / sum;
            }
        }


        //Computing \vartheta
        double[] qzz = new double[KK];
        double maxQzz = Double.NEGATIVE_INFINITY;
        for (int kk = 0; kk < KK; kk++)
        {
            qzz[kk] = 0;
            for (int ii = 0; ii < contData.size(); ii++)
            {
                sum = 0;
                for (int tt = 0; tt < TT; tt++)
                    sum += unormalizedQtt[ii][kk][tt];
                qzz[kk] += (maxQtt[ii][kk] + FastMath.log(sum)) * contMap.get(contData.get(ii));// multiply number of occurences of token
            }

            qzz[kk] += EtopStick[kk] + cxtEloglik[kk];
            if (qzz[kk] > maxQzz)
                maxQzz = qzz[kk];
        }

        // Convert to expential
        sum = 0;
        for (int kk = 0; kk < KK; kk++)
        {
            qzz[kk] = FastMath.exp(qzz[kk] - maxQzz);
            sum += qzz[kk];
        }
        //Normalize
        for (int kk = 0; kk < KK; kk++)
        {
            qzz[kk] = qzz[kk] / sum;
        }


        //region Update intermediate params \chihat, \rhoshat,\zetashat, \varphishat, \alphashat


        //Update \chihat
        double[][][] alphas = new double[KK][TT][MM];

        //Computing alphas(:,:,_MM-1)
        for (int kk = 0; kk < KK; kk++)
        {
            for (int tt = 0; tt < TT; tt++)
            {
                sum = 0;
                for (int ii = 0; ii < contData.size(); ii++)
                    sum += qtt[ii][kk][tt] * contEloglik[ii][MM - 1] * contMap.get(contData.get(ii)); // multiply number of occurences of token
                alphas[kk][tt][MM - 1] = qzz[kk] * sum;
            }
        }

        //Computing alphas(:,:,0:_MM-2) and chishat

        for (int kk = 0; kk < KK; kk++)
        {
            for (int tt = 0; tt < TT; tt++)
            {
                for (int mm = 0; mm < MM - 1; mm++)
                {
                    sum = 0;
                    for (int ii = 0; ii < contData.size(); ii++)
                        sum += qtt[ii][kk][tt] * contEloglik[ii][mm] * contMap.get(contData.get(ii)); // multiply number of occurences of token;
                    alphas[kk][tt][mm] = qzz[kk] * sum;

                    result.docChishat[kk][tt][mm] += Esharestick[mm] - Esharestick[MM - 1] + ngroups * (alphas[kk][tt][mm] - alphas[kk][tt][MM - 1]);
                }
            }
        }

        //Update \rhoshat
        sum = qzz[KK - 1];
        for (int kk = KK - 2; kk >= 0; kk--)
        {
            result.docRhoshat[kk][0] += 1 + ngroups * qzz[kk];
            result.docRhoshat[kk][1] += gg + ngroups * sum;
            sum += qzz[kk];

        }

        //Update \zetashat
        for (int kk = 0; kk < KK; kk++)
        {
            // Computing the last TT
            double[] sumzetas = new double[TT];
            for (int ii = 0; ii < contData.size(); ii++)
                for (int tt = 0; tt < TT; tt++)
                {
                    sumzetas[tt] += qtt[ii][kk][tt] * contMap.get(contData.get(ii)); // multiply number of occurences of token;;
                }
            sum = sumzetas[TT - 1];
            for (int tt = TT - 2; tt >= 0; tt--)
            {
                result.docZetashat[kk][tt][0] += 1 + ngroups * qzz[kk] * sumzetas[tt];
                result.docZetashat[kk][tt][1] += aa + ngroups * qzz[kk] * sum;
                sum += sumzetas[tt];

            }
        }

        //Update \varphishat
        double[] sumkt = new double[MM];

        for (int mm = 0; mm < MM; mm++)
        {
            for (int kk = 0; kk < KK; kk++)
            {
                for (int tt = 0; tt < TT; tt++)
                {
                    sumkt[mm] += qcc[kk][tt][mm];
                }
            }
        }
        sum = sumkt[MM - 1];
        for (int mm = MM - 2; mm >= 0; mm--)
        {

            result.docVarphishat[mm][0] += 1 +  sumkt[mm];
            result.docVarphishat[mm][1] += ee + sum;
//            result.docVarphishat[mm][0] += 1 + ngroups * sumkt[mm];
//            result.docVarphishat[mm][1] += ee + ngroups * sum;
            sum += sumkt[mm];

        }

        // Update \lambdashat
        double[][] dataProb = new double[contData.size()][MM];
        for (int ii = 0; ii < contData.size(); ii++)
            for (int mm = 0; mm < MM; mm++)
            {
                for (int kk = 0; kk < KK; kk++)
                    for (int tt = 0; tt < TT; tt++)
                    {
                        dataProb[ii][mm] += qtt[ii][kk][tt] * qcc[kk][tt][mm] * qzz[kk];
                    }
                try {
                    result.docLambdahat.get(mm).add(contData.get(ii), ngroups / batchCount * dataProb[ii][mm] * contMap.get(contData.get(ii))); // multiply number of occurences of token
                }catch (Exception e){
                    e.printStackTrace();
                }
            }

        //Updating \alphashat
        try {
            for (int kk = 0; kk < KK; kk++) {
                result.docAlphahat.get(kk).add(xx.get(jj), ngroups / batchCount * qzz[kk]);

            }
        }catch (Exception e){
            e.printStackTrace();
        }

        // Computing running time
        innerTimer.stop();
        System.out.println("\tFinished document "+jj+ " elapse="+ innerTimer.getElaspedSeconds());
        return result;
    }

    public HashMap<Object, Integer> GetUniqueValue(ArrayList<Object> values)
    {
        HashMap<Object, Integer> uniqueValues = new HashMap<Object, Integer>();

        for (int ii = 0; ii < values.size(); ii++)
        {
            Integer val=uniqueValues.get(values.get(ii));
            if (val==null)
                uniqueValues.put(values.get(ii), 1);
            else
                uniqueValues.put(values.get(ii),val+1);
        }
        return uniqueValues;

    }

    public static void smallDataExp()
    {
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

        MC2StochasticVariationalInference mc2 = new MC2StochasticVariationalInference(trunK, trunT, trunM, ee, aa, vv, L, H);
        mc2.loadData(xx, ss);
        mc2.initialize();
        String strOutMat = "D:\\";
//        mc2.sviOptimizer(2, 1, 1, 0.6, strOutMat);
        mc2.sviCategoryOptimizer(2, 2, 1, 0.6, strOutMat);

        //mc2.SVIParallelOptimizer(1, 2, 1, .6, strOutMat);
        //wsndp.CollapsedGibbsInference(initK, initM, 10, 1000, 1, stirlingFilename, strOutMat, cp, smooth, stirlingFolder);
        //System.Console.WriteLine(Environment.CurrentDirectory.ToString());
    }
    public static void NIPSExperiment_AuthorContext()
    {



        //string folder = @"P:\WithVu_Experiment\csdp shared\csdp shared NIPS dataset\splitedmatfiles_wsndp\";
        String strMat = "F:\\MyPhD\\Code\\CSharp\\MC2_VU_CODE\\csdp shared\\csdp shared NIPS dataset\\nips_wsndp_Csharp_author_90pctrain.mat";

        MC2InputDataMultCat data= MatlabJavaConverter.readMC2DataFromMatFiles("V","VAuthor", strMat);

        System.out.println("Vocabulary Size = " + data.contentDim);

        System.out.println("Author Vocabulary Size = " +data.contextDim);

        System.out.println("Number of Documents = " + data.ngroups);



        // base measure of topic
        double sym = 0.01;
        int trunM = 100;
        MultinomialDirichlet H = new MultinomialDirichlet(data.contentDim, sym * data.contextDim);

        // base measure of author
        double symAuthor = 0.01;
        int trunK = 20;
        int trunT = 50;
        MultinomialDirichlet L = new MultinomialDirichlet(data.contextDim, symAuthor * data.contextDim);


        double aa = 10;
        double ee = 10;
        double vv = 10;

        System.out.println("Creating SVI inference engine...");

        MC2StochasticVariationalInference mc2 = new MC2StochasticVariationalInference(trunK, trunT, trunM, ee, aa, vv, L, H);

        mc2.loadData(data.xx, data.ss);

        System.out.println("Initializing ...");
        mc2.initialize();
        // string strOutMat = @"D:\Nips_Author_SVI_CSharp_Parallel\";
        String strOutMat = "D:\\Nips_Author_SVI_CSharp_Serial\\";

        // SVI params
        int numIter = 2;
        int batchSize = 50;
        double varrho = 1;
        double iota = 0.6;

        System.out.println("Running ...");
        mc2.sviCategoryOptimizer(numIter, batchSize, varrho, iota, strOutMat);
    }
}
