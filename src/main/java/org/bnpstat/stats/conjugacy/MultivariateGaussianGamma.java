package org.bnpstat.stats.conjugacy;

import org.apache.commons.math.special.Gamma;
import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Implement of (multivariate) Gaussian-Gamma compound model with fixed variance
 * theta ~ Gaussian(mean,I/(s0*precision)); precision ~ Gamma(a0,b0)
 * observation (xx) ~ Gaussian(theta,1/precision)
 */
public class MultivariateGaussianGamma implements BayesianComponent {

    //Dimension
    int dim;
    // Mean for prior
    private double[] mean;
    // Gamma parameters for  precision prior
    private double a0;
    private double b0;
    // Prior precision scale
    private double s0;

    // Mean for posterior
    private double[] postMean;
    // Gamma parameters for  precision prior
    private double an;
    private double bn;
    // Posterior precision scale
    private double sn;

    // Fixed variance for observed data
    //  private double obsVar;
    // Sufficient statistics
    private double[] suffStatsMean;
    private double suffStatsVar;
    private double numItems;


    /**
     * The constructor
     * @param mu0: The prior mean
     * @param s0: The prior precision scale
     * @param a0: The first parameter for precision prior
     * @param b0: The second parameter for precision prior
     */
    public MultivariateGaussianGamma(double[] mu0, double s0, double a0, double b0){
        // implicit data
        numItems=0;
        dim=mu0.length;
        //prior
        mean=mu0;
        this.a0=a0;
        this.b0=b0;
        this.s0=s0;

        //posterior
        postMean= Arrays.copyOf(mu0,mu0.length);
        this.an=a0;
        this.bn=b0;
        this.sn=s0;

        // Sufficient statistic for computing posterior
        suffStatsMean=new double[dim];
        suffStatsVar=0;
    }
    @Override
    public void add(Object observation) throws Exception {
        if (observation!=null) {
            if (DoubleMultivariateData.class.isInstance(observation)) {
                DoubleMultivariateData obs = (DoubleMultivariateData) observation;
                numItems++;
                for (int i = 0; i < dim; i++) {
                    double temp = obs.get(i);
                    suffStatsMean[i] += temp;
                    suffStatsVar += temp * temp;
                }
            } else
                throw new Exception("Data item type is not supported. Used Integer or Double object instead");
            // Update the posterior
            sn = s0 + numItems;
            double mu0sq = 0;
            double munsq = 0;

            for (int i = 0; i < dim; i++) {
                postMean[i] = (s0 * mean[i] + suffStatsMean[i]) / sn;
                mu0sq += mean[i] * mean[i];
                munsq += postMean[i] * postMean[i];
            }
            an = a0 + dim * numItems / 2;
            bn = b0 + (suffStatsVar + s0 * mu0sq - sn * munsq) / 2;
        }
    }

    @Override
    public void add(Object observation, double prob) throws Exception {
        if (observation!=null) {
            if (DoubleMultivariateData.class.isInstance(observation)) {
                DoubleMultivariateData obs = (DoubleMultivariateData) observation;
                numItems += prob;
                for (int i = 0; i < dim; i++) {
                    double temp = obs.get(i);
                    suffStatsMean[i] += temp * prob;
                    suffStatsVar += temp * temp * prob;
                }
            } else
                throw new Exception("Data item type is not supported. Used Integer or Double object instead");
            // Update the posterior
            sn = s0 + numItems;
            double mu0sq = 0;
            double munsq = 0;

            for (int i = 0; i < dim; i++) {
                postMean[i] = (s0 * mean[i] + suffStatsMean[i]) / sn;
                mu0sq += mean[i] * mean[i];
                munsq += postMean[i] * postMean[i];
            }
            an = a0 + dim * numItems/2;
            bn = b0 + (suffStatsVar + s0 * mu0sq - sn * munsq) / 2;
        }
    }

    @Override
    public void addRange(Object[] observationArray) throws Exception {
        for (int ii=0;ii<observationArray.length;ii++){
            add(observationArray[ii]);
        }
    }

    @Override
    public void remove(Object observation) throws Exception {
        if (observation!=null) {
            if (numItems < 1)
                throw new Exception("There is no more data in the component to delete");
            else if (numItems == 1) {
                postMean = Arrays.copyOf(mean, mean.length);
                this.an = a0;
                this.bn = b0;
                this.sn = s0;
                this.suffStatsVar = 0;
                suffStatsMean = new double[dim];
            } else {
                if (DoubleMultivariateData.class.isInstance(observation)) {
                    DoubleMultivariateData obs = (DoubleMultivariateData) observation;
                    for (int i = 0; i < dim; i++) {
                        double temp = obs.get(i);
                        suffStatsMean[i] -= temp;
                        suffStatsVar -= temp * temp;
                    }
                    numItems -= 1;
                    // Update the posterior
                    sn = s0 + numItems;
                    double mu0sq = 0;
                    double munsq = 0;

                    for (int i = 0; i < dim; i++) {
                        postMean[i] = (s0 * mean[i] + suffStatsMean[i]) / sn;
                        mu0sq += mean[i] * mean[i];
                        munsq += postMean[i] * postMean[i];
                    }
                    an = a0 + dim * numItems/2;
                    bn = b0 + (suffStatsVar + s0 * mu0sq - sn * munsq) / 2;
                } else
                    throw new Exception("Data item type is not supported. Used Integer or Double object instead");

            }
        }
    }

    @Override
    public Object getPosterior() throws Exception {
        throw new Exception("The method is not implemented yet!");
    }

    @Override
    public double logPredictive(Object observation) throws Exception {
        if (observation!=null) {
            double[] val;
            if (DoubleMultivariateData.class.isInstance(observation)) {
                DoubleMultivariateData obs = (DoubleMultivariateData) observation;
                val = obs.getVal();
            } else
                throw new Exception("Data item type is not supported. Used Integer or Double object instead");
            double[][] covar = new double[dim][dim];
            for (int i = 0; i < dim; i++)
                covar[i][i] = sn * bn / an;
            return Math.log((new MultivariateNormalDistribution(postMean, covar)).density(val));
        }
        else return 0; // missing data point
    }

    @Override
    public double logMarginal() throws Exception{
        throw new Exception("The method is not implemented yet!");

    }

    @Override
    public double expectationLogLikelihood(Object observation) throws Exception {
        if (observation != null) {
            double[] xx;
            if (DoubleMultivariateData.class.isInstance(observation)) {
                DoubleMultivariateData obs = (DoubleMultivariateData) observation;
                xx = obs.getVal();
            } else
                throw new Exception("Data item type is not supported. Used Integer or Double object instead");
            double val = 0;
            double mu0sq = 0;
            double munsq = 0;

            for (int i = 0; i < dim; i++) {
                val += xx[i] * an / bn * postMean[i];
                mu0sq += xx[i] * xx[i];
                munsq += postMean[i] * postMean[i];
            }
            val -= an / (2 * bn) * mu0sq;
//            val -= (an * munsq / bn + dim / sn - dim * (dig(an) - Math.log(bn)) + dim * Math.log(2 * Math.PI)) / 2;
            val -= (an * munsq / bn + dim / sn - dim * (Gamma.digamma(an) - Math.log(bn)) + dim * Math.log(2 * Math.PI)) / 2;

            return val;
        }
        else return 0; // missing data point
    }
    // This for compute expectationLogLikelihood of multiple obs w.r.t multiple atoms
    public static double[][] expectationLogLikelihoodGeneric(ArrayList<Object> observations, ArrayList<BayesianComponent> pp)  throws Exception {
        int nCount = observations.size();
        int nDists = pp.size();
        double[][] Ell=new double[nCount][nDists];
        for (int ii = 0; ii < nCount; ii++)
        {
            for (int jj = 0; jj < nDists; jj++) {
                Ell[ii][jj]=pp.get(jj).expectationLogLikelihood(observations.get(ii));
            }
        }
        return Ell;
    }

    @Override
    public void stochasticUpdate(BayesianComponent pp, double kappa) throws Exception {
        double tempan=(1-kappa)*getPostAlpha()+kappa* ((MultivariateGaussianGamma)pp).getPostAlpha();
        double tempsn=(1-kappa)*getPostLambda()+kappa* ((MultivariateGaussianGamma)pp).getPostLambda();
        double musqOld=0;
        double musqNew=0;
        double musqUpdate=0;
        for (int i=0;i<dim;i++){
            musqOld+=postMean[i]*postMean[i];
            double temp=((MultivariateGaussianGamma)pp).postMean[i];
            musqNew+=temp*temp;
            postMean[i]=(1-kappa)*getPostLambda()*postMean[i]+kappa*((MultivariateGaussianGamma)pp).getPostLambda()*temp;
            postMean[i]/=tempsn;
            musqUpdate+= postMean[i]* postMean[i];
        }
        bn=(1-kappa)*getPostBeta()+kappa*((MultivariateGaussianGamma)pp).getPostBeta();
        bn+=((1-kappa)*getPostLambda()*musqOld+kappa*((MultivariateGaussianGamma)pp).getPostLambda()*musqNew-tempsn*musqUpdate)/2;
        an=tempan;
        sn=tempsn;
//        double newVar=1/((1-kappa)/postVariance+kappa/((MultivariateGaussianGamma)pp).getPostVariance());
//        double newMean=newVar*((1-kappa)*postMean/postVariance+kappa*((GaussianGaussian)pp).getPostMean()/((GaussianGaussian)pp).getPostVariance());
    }

    @Override
    public void plus(BayesianComponent pp) throws Exception {

        numItems=getNumItems()+((MultivariateGaussianGamma)pp).getNumItems();


        an=a0+numItems*dim/2;
        sn=s0+numItems;

        double mu0sq=0;
        double munsq=0;
        for(int i=0;i<dim; i++){
            suffStatsMean[i]+=((MultivariateGaussianGamma)pp).suffStatsMean[i];
            postMean[i]=(s0*mean[i]+suffStatsMean[i])/sn;
            mu0sq+=mean[i]*mean[i];
            munsq+=postMean[i]*postMean[i];
            // Update mean sufficient statistics
        }
        // Update variance sufficient statistics
        suffStatsVar+=((MultivariateGaussianGamma)pp).suffStatsVar;
        bn=b0+(suffStatsVar+s0*mu0sq-sn*munsq)/2;
    }
    public Object clone(){
        MultivariateGaussianGamma out= new MultivariateGaussianGamma(mean,s0,a0,b0);
        out.setNumItems(numItems);
        out.setPostMean(postMean);
        out.setAn(an);
        out.setBn(bn);
        out.setSn(sn);
        out.setSuffStatsMean(suffStatsMean);
        out.setSuffStatsVar(suffStatsVar);
        return out;
    }
    public double getNumItems() {
        return numItems;
    }

    public void setNumItems(double numItems) {
        this.numItems = numItems;
    }

    public double[] getPostMean(){
        return postMean;
    }
    public void setPostMean(double[] postMean) {
        this.postMean = Arrays.copyOf(postMean,postMean.length);
    }

    public double getAn(){
        return an;
    }
    public double getPostAlpha(){
        return getAn();
    }

    public void setAn(double an) {
        this.an = an;
    }

    public double getBn() {
        return bn;
    }
    public double getPostBeta(){
        return getBn();
    }
    public void setBn(double bn) {
        this.bn = bn;
    }

    public double getSn() {
        return sn;
    }
    public double getPostLambda(){
        return getSn();
    }

    public void setSn(double sn) {
        this.sn = sn;
    }

    public int getDim() {
        return dim;
    }

    public double getA0() {
        return a0;
    }

    public double getS0() {
        return s0;
    }

    public double getB0() {
        return b0;
    }

    public double[] getSuffStatsMean(){
        return suffStatsMean;
    }
    public void setSuffStatsMean(double[] suffStatsMean) {
        this.suffStatsMean = Arrays.copyOf(suffStatsMean,suffStatsMean.length);
    }

    public void setSuffStatsVar(double suffStatsVar) {
        this.suffStatsVar = suffStatsVar;
    }

    @Override
    public String toString() {
        return "MultivariateGaussianGamma: \n\tPrior:\t\t alpha="+a0+", beta="+b0+", lambda="+s0+"\t mean="+Arrays.toString(mean)+
                "\n\tPosterior:\t alpha="+an+", beta="+bn+", lambda="+sn+"\t mean="+Arrays.toString(postMean)+"\n";
    }
    private double dig(double xx) {
        double x = xx;
        double r = 0.0;

        while (x<=5) {
            r -= 1/x;
            x += 1;
        }

        double f = 1d/(x * x);
        double t = f*(-1/12.0 +
                f*(1/120.0 +
                        f*(-1/252.0 +
                                f*(1/240.0 +
                                        f*(-1/132.0 +
                                                f*(691/32760.0 +
                                                        f*(-1/12.0 +
                                                                f*3617.0/8160.0)))))));
       return r + Math.log(x) - 0.5/x + t;
    }
}

