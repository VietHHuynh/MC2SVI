package org.bnpstat.stats.conjugacy;

import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.distribution.NormalDistribution;
/**
 * Implement of (univariate) Gaussian-Gaussian compound model with fixed variance
 * \theta ~ Gaussian(mean,variance);
 * observation ~ Gaussian(mm,\theta,obsVar)
 */
public class GaussianGaussian implements BayesianComponent {
    private double numItems;

    // Mean for prior
    private double mean;
    // Mean for posterior
    private double postMean;
    // Variance for prior
    private double variance;
    // Variance for posterior
    private double postVariance;
    // Fixed variance for observed data
    private double obsVar;
    // Sufficient statistics
    private double suffStats;


    /**
     * The constructor
     * @param mu0: The prior mean
     * @param var0: The prior variance
     * @param fixVar: The fixed variance for observation
     */
    public GaussianGaussian(double mu0, double var0, double fixVar){
        numItems=0;
        mean=mu0;
        variance=var0;
        obsVar=fixVar;
        postMean=mu0;
        postVariance=var0;
    }
    @Override
    public void add(Object observation) throws Exception {
        if (Integer.class.isInstance(observation) ) {
            numItems++;
            suffStats +=((Integer)observation).intValue();
        }else if( Double.class.isInstance(observation)){
            numItems++;
            suffStats +=((Double)observation).doubleValue();
        }else
            throw new Exception("Data item type is not supported. Used Integer or Double object instead");
        // Update the posterior
        postVariance=obsVar*variance/(numItems*variance+obsVar);
        postMean=postVariance*(mean/variance+suffStats/obsVar);
    }

    @Override
    public void add(Object observation, double prob) throws Exception {
        if (Integer.class.isInstance(observation) ) {
            numItems+=prob;
            suffStats +=((Integer)observation).intValue()*prob;
        }else if( Double.class.isInstance(observation)){
            numItems+=prob;
            suffStats +=((Double)observation).doubleValue()*prob;
        }else
            throw new Exception("Data item type is not supported. Used Integer or Double object instead");
        // Update the posterior
        postVariance=obsVar*variance/(numItems*variance+obsVar);
        postMean=postVariance*(mean/variance+suffStats/obsVar);
    }

    @Override
    public void addRange(Object[] observationArray) throws Exception {
        for (int ii=0;ii<observationArray.length;ii++){
            add(observationArray[ii]);
        }
    }

    @Override
    public void remove(Object observation) throws Exception {

        if(numItems<1)
            throw new Exception("There is no more data in the component to delete");
        else if (numItems==1) {
            postMean = mean;
            postVariance = variance;
        }
        else{
            if (Integer.class.isInstance(observation) ) {
                postMean=postMean*(numItems/(numItems-1)-((Integer)observation).intValue()/(numItems-1));
            }else if( Double.class.isInstance(observation)) {
                postMean = postMean * (numItems / (numItems - 1) - ((Double) observation).doubleValue() / (numItems - 1));
            } else
                throw new Exception("Data item type is not supported. Used Integer or Double object instead");
            numItems-=1;
        }
    }

    @Override
    public Object getPosterior() throws Exception {
        throw new Exception("The method is not implemented yet!");
    }

    @Override
    public double logPredictive(Object observation) throws Exception {
        double val;
        if (Integer.class.isInstance(observation) ) {
            val=((Integer)observation).intValue();
        }else if( Double.class.isInstance(observation)) {
            val= ((Double) observation).doubleValue();
        } else
            throw new Exception("Data item type is not supported. Used Integer or Double object instead");
        return (new NormalDistribution(postMean,postVariance)).logDensity(val);
    }

    @Override
    public double logMarginal() throws Exception{
        throw new Exception("The method is not implemented yet!");

    }

    @Override
    public double expectationLogLikelihood(Object observation) throws Exception {
        double xx;
        if (Integer.class.isInstance(observation) ) {
            xx=((Integer)observation).intValue();
        }else if( Double.class.isInstance(observation)) {
            xx= ((Double) observation).doubleValue();
        } else
            throw new Exception("Data item type is not supported. Used Integer or Double object instead");
        double val=(xx*postMean-xx*xx/2)/obsVar;
        val-=(postMean*postMean/obsVar+postVariance/obsVar+Math.log(2*Math.PI))/2;
        return val;
    }

    @Override
    public void stochasticUpdate(BayesianComponent pp, double kappa) throws Exception {
        double newVar=1/((1-kappa)/postVariance+kappa/((GaussianGaussian)pp).getPostVariance());
        double newMean=newVar*((1-kappa)*postMean/postVariance+kappa*((GaussianGaussian)pp).getPostMean()/((GaussianGaussian)pp).getPostVariance());
    }

    @Override
    public void plus(BayesianComponent pp) throws Exception {

        postVariance=1/(1/postVariance+1/((GaussianGaussian)pp).getPostVariance()-1/variance);
        postMean=postVariance*(mean/variance+(suffStats+((GaussianGaussian)pp).getSuffStats())/obsVar);
        suffStats+=((GaussianGaussian)pp).getSuffStats();
        numItems+=((GaussianGaussian)pp).getNumItems();

    }
    public Object clone(){
        GaussianGaussian out= new GaussianGaussian(mean,variance,obsVar);
        out.setNumItems(numItems);
        out.setPostMean(postVariance);
        out.setPostVariance(postVariance);
        out.setSuffStats(suffStats);
        return out;
    }
    public double getNumItems() {
        return numItems;
    }

    public void setNumItems(double numItems) {
        this.numItems = numItems;
    }

    public double getPostMean(){
        return postMean;
    }
    public void setPostMean(double postMean) {
        this.postMean = postMean;
    }

    public double getPostVariance(){
        return postVariance;
    }

    public void setPostVariance(double postVariance) {
        this.postVariance = postVariance;
    }

    public double getSuffStats(){
        return suffStats;
    }
    public void setSuffStats(double suffStats) {
        this.suffStats = suffStats;
    }

    @Override
    public String toString() {
        return "Prior: mean="+mean+", variance="+variance+", observation variance="+obsVar+"\nPosterior: mean="+postMean+", variance="+postVariance+", sufficient statistics="+suffStats+"\n";
    }
}
