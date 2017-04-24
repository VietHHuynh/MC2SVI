package org.bnpstat.stats.conjugacy;

/**
 * Created by hvhuynh on 11/6/2015.
 */
import org.apache.commons.math3.analysis.function.Log;
import org.bnpstat.maths.SparseVector;
import org.apache.commons.math3.special.Gamma;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Implement of Multinomial-Dirichlet compound model where *symmetric* Dirichlet is assumed
 *  pi ~ Dirichlet(alpha/dim, ...., alpha/dim);
 *  observation ~ Mult(mm,pi)
 */
public class MultinomialDirichlet implements BayesianComponent {

    private double numItems;
    // dim = dimensionality
    private int dim;

    // aa = parameter for the symmetric Dirichlet
    private double aa;


    // mi = individual count (dx1)
    private SparseVector mi;


    // mm = total counts (sum over all elements in xx)
    private double mm;

    // normalized constant
    private double Z0;

    public MultinomialDirichlet(int dd, double aa)
    {
        this.dim = dd;
        this.aa = aa / (double)dd;
        this.mi = new SparseVector(dd);
        this.mm = 0;
        this.numItems=0;
        this.Z0 = 0;
    }

    public MultinomialDirichlet(int dd, double aa, SparseVector mi, long mm)
    {
        this.dim = dd;
        this.aa = aa / (double)dd;
        this.mi = mi;
        this.mm = mm;
        this.numItems=0;
        this.Z0 = 0;
    }
    public double getNumItems() {
        return numItems;
    }
    public void setNumItems(int value){
        numItems=value;
    }
    public int getDim(){return dim;}

    /**
     *  Add a new data item into this component.
     * @param observation - could be a single draw or a (sparse) vector of count
     * @throws Exception  - IllegalArgumentException if  wrong data types
     */
    public void add(Object observation) throws Exception{
// observation is a single draw of x
        if (Integer.class.isInstance(observation) )
        {
            int xx = ((Integer)observation).intValue();
            if (0 <= xx && xx < this.dim)
            {
                this.numItems++;
                this.mi.addValue(xx,1);
                this.mm++;
            }
            else
                throw new Exception("Adding a data input out of bound");
        }
        // observation is a full vector of count i.e., the value at v-th element is
        // the number of times v-th word were observed
        else if (observation instanceof int[] )
        {
            int[] x = (int[])observation;
            if (x.length != this.dim)
                throw new IllegalArgumentException("Size of count vector does not match");
            this.numItems++;

            long m = 0;
            double lgsum = 0; // \sum_{i=1}^V \log\Gamma(_aa[i] + _mi[i])
            for (int i = 0; i < this.dim; i++)
            {
                if (x[i] > 0)
                {
                    this.mi.addValue(i,x[i]);
                    m += x[i];
                    lgsum += Gamma.logGamma(x[i] + 1);
                }
            }

            this.mm += m;
            this.Z0 += Gamma.logGamma(m + 1) - lgsum;
        }
        else if (observation instanceof SparseVector)
        {
            SparseVector x = (SparseVector)observation;
            if(x.getNonzeoLength()==0) return; // The missing value

            if (x.getLength()!= this.dim)
                throw new IllegalArgumentException("Size of count vector does not match");

            this.numItems++;

            long totalCount = 0;
            double sumlog = 0; // \sum_{i=1}^V \log\Gamma(_aa[i] + _mi[i])

           Integer[] index=x.getIndices();

            for (int i = 0; i < index.length; i++) {
                int value=(int)x.getValue(index[i]);
                this.mi.addValue(index[i],value);
                totalCount += value;
                sumlog += Gamma.logGamma(value + 1);
            }
            this.mm += totalCount;
            this.Z0 += Gamma.logGamma(totalCount + 1) - sumlog;
        }
        else
        throw new Exception("Data item type is not supported");
    }

    /**
     * Add a new data item into this component with a probability.
     * @param observation - the observation to be added - it has to be cast into proper datatype
     *              depending on data we are working with
     * @param prob - the probability observation belong to this component
     */
    public void add(Object observation, double prob) throws Exception{
        if (Integer.class.isInstance(observation) )
        {
            int xx = ((Integer)observation).intValue();
            if (0 <= xx && xx < this.dim)
            {
                this.numItems+=prob;
                this.mi.addValue(xx,prob);
                this.mm+=prob;
            }
            else
                throw new Exception("Adding a data input out of bound");
        }
        // observation is a full vector of count i.e., the value at v-th element is
        // the number of times v-th word were observed
        else if (observation instanceof int[] )
        {
            int[] x = (int[])observation;
            if (x.length != this.dim)
                throw new IllegalArgumentException("Size of count vector does not match");
            this.numItems+=prob;

            long m = 0;
            double lgsum = 0; // \sum_{i=1}^V \log\Gamma(_aa[i] + _mi[i])
            for (int i = 0; i < this.dim; i++)
            {
                if (x[i] > 0)
                {
                    this.mi.addValue(i,x[i]*prob);
                    m += x[i]*prob;
                    lgsum += Gamma.logGamma(x[i]*prob + 1);
                }
            }

            this.mm += m;
            this.Z0 += Gamma.logGamma(m + 1) - lgsum;
        }
        else if (observation instanceof SparseVector)
        {
            SparseVector x = (SparseVector)observation;
            if(x.getNonzeoLength()==0) return; // The missing value
            if (x.getLength()!= this.dim)
                throw new IllegalArgumentException("Size of count vector doesnot match");

            this.numItems+=prob;

            double totalCount = 0;
            double sumlog = 0; // \sum_{i=1}^V \log\Gamma(_aa[i] + _mi[i])

            Integer[] index=x.getIndices();

            for (int i = 0; i < index.length; i++) {
                int value=(int)x.getValue(index[i]);
                this.mi.addValue(index[i],value*prob);
                totalCount += value*prob;
                sumlog += Gamma.logGamma(value*prob + 1);
            }


            this.mm += totalCount;
            this.Z0 += Gamma.logGamma(totalCount + 1) - sumlog;
        }
        else
            throw new Exception("Data item type is not supported");
    }

    public void addRange(Object[] observationArray) throws Exception     {
        for (int i = 0; i < observationArray.length; i++)
            add(observationArray[i]);
    }

    public void remove(Object observation) throws Exception {
        if (Integer.class.isInstance(observation))
        {
            int xx =  ((Integer)observation).intValue();
            if (0 <= xx && xx < this.dim)
            {
                this.numItems--;
                this.mi.subValue(xx,1);
                this.mm--;
            }
            else
                throw new Exception("Deleting an data item out of bound");
        }
        else if (observation instanceof int[])
        {
            int[] countVect = (int[])observation;
            if (countVect.length != this.dim)
                throw new Exception("Size of count vector doesnot match");

            this.numItems--;

            long m = 0;
            double lgsum = 0;
            for (int i = 0; i < this.dim; i++)
            {
                if (countVect[i] > 0)
                {
                    this.mi.subValue(i,countVect[i]);
                    m += countVect[i];
                    lgsum += Gamma.logGamma(countVect[i] + 1);
                }
            }

            this.mm -= m;
            //this._Z0 -= Function.LogGamma(m + 1) + lgsum;
            this.Z0 = this.Z0 - Gamma.logGamma(m + 1) + lgsum;
        }
        else if (observation instanceof SparseVector)
        {
            SparseVector x = (SparseVector)observation;
            if(x.getNonzeoLength()==0) return; // The missing value

            if (x.getLength() != this.dim)
                throw new Exception("Size of count vector doesn ot match");

            this.numItems--;

            long m = 0;
            double lgsum = 0;

            Integer[] index=x.getIndices();

            for (int i = 0; i < index.length; i++) {
                int value=(int)x.getValue(index[i]);
                this.mi.addValue(index[i],value);
                m += value;
                lgsum += Gamma.logGamma(value + 1);
            }


            this.mm -= m;
            //this._Z0 -= Function.LogGamma(m + 1) + lgsum;// error here
            this.Z0 = this.Z0 - Gamma.logGamma(m + 1) + lgsum;
        }
        else
        throw new Exception("Data item type is not supported");
    }

    public Object getPosterior() throws Exception {
        throw new Exception("The method is not implemented yet!");
    }

    public double logPredictive(Object observation) throws Exception{

        double ll ;
        Log lg= new Log();

        // observation is a single draw of x
        if (Integer.class.isInstance(observation) )
        {
            int xx = ((Integer)observation).intValue();
            if (0 <= xx && xx < this.dim)
            {
                ll = lg.value((aa + mi.getValue(xx)) / (aa * dim + mm));
            }
            else
                throw new Exception("Adding a data input out of bound");
        }
        // observation is a full vector of count i.e., the value at v-th element is
        // the number of times v-th word were observed
        else if (observation instanceof int[])
        {
            int[] x = (int[])observation;
            if (x.length != this.dim)
                throw new Exception("Size of count vector does not match");

            long sumMM = 0;
            double lgsum1 = 0; // \sum_{i=1}^V \log \Gamma(x[i] + 1)
            double lgsum2 = 0; // \sum_{i=1}^V {\log\Gamma(_aa[i] + _mi[i] + x[i]) - \log\Gamma(_aa[i] + _mi[i])}

            for (int i = 0; i < x.length; i++) {
                if (x[i] > 0) {
                    sumMM += x[i];
                    lgsum1 += Gamma.logGamma(x[i] + 1);
                    lgsum2 += Gamma.logGamma(aa + mi.getValue(i) + x[i]) - Gamma.logGamma(aa + mi.getValue(i));
                }
            }

            ll = Gamma.logGamma(sumMM + 1) - lgsum1
                    + Gamma.logGamma(aa * dim + mm) - Gamma.logGamma(aa *dim + mm + sumMM)
                    + lgsum2;
        }
        else if (observation instanceof SparseVector)
        {
            SparseVector x = (SparseVector)observation;
            if (x.getNonzeoLength() == 0)
                return 0;

            if (x.getLength() != this.dim)
                throw new Exception("Size of count vector does not match: data size ("+x.getLength()+")!= distribution dimension ("+this.dim+")");

            long sumMM = 0;
            double lgsum1 = 0; // \sum_{i=1}^V \log \Gamma(x[i] + 1)
            double lgsum2 = 0; // \sum_{i=1}^V {\log\Gamma(_aa[i] + _mi[i] + x[i]) - \log\Gamma(_aa[i] + _mi[i])}

            //for (int i = 0; i < x.Length; i++)
            //{
            //    mm += x[i];
            //    lgsum1 += Function.LogGamma(x[i] + 1);
            //    lgsum2 += Function.LogGamma(_gamma + _mi[i] + x[i]) - Function.LogGamma(_gamma + _mi[i]);
            //}
            Integer[] index=x.getIndices();

            for (int i = 0; i < index.length; i++) {


                int value=(int)x.getValue(index[i]);
                sumMM += value;
                lgsum1 += Gamma.logGamma(value + 1);
                lgsum2 += Gamma.logGamma(aa + mi.getValue(i) + value) - Gamma.logGamma(aa + mi.getValue(i));
            }

            ll = Gamma.logGamma(sumMM + 1) - lgsum1
                    + Gamma.logGamma(aa * dim + mm) - Gamma.logGamma(aa * dim + mm + sumMM)// error _mm => .... change ll
                    + lgsum2;
        }
        else
        throw new Exception("Data item type is not supported");

        return ll;
    }

    public double logMarginal() {
        double ll = 0;

        double lgsum1 = 0; // \sum_{i=1}^V \log\Gamma(_aa[i] + _mi[i])
        double lgsum2 = 0; // \sum_{i=1}^V \log\Gamma(_aa[i])
        for (int i = 0; i < dim; i++)
        {
            lgsum1 += Gamma.logGamma(aa + mi.getValue(i));
            lgsum2 += Gamma.logGamma(aa);
        }

        ll = Z0 + Gamma.logGamma(aa * dim) - Gamma.logGamma(aa * dim + mm)
                + lgsum1 - lgsum2;
        return ll;
    }

    public double expectationLogLikelihood(Object observation)  throws Exception{

        double ll = 0;
        double postParamSum = 0;
        double[] postParam = new double[this.dim];

        for (int i = 0; i < this.dim; i++)
        {
            postParam[i] = this.aa + this.mi.getValue(i);
            postParamSum += postParam[i];
        }

        // observation is a single draw of x
        if (Integer.class.isInstance(observation) )
        {
            int v = ((Integer)observation).intValue();
            if (0 <= v && v < this.dim)
            {
                ll =Gamma.digamma(postParam[v]) - Gamma.digamma(postParamSum);
            }
            else
                throw new Exception("Data input out of bound");
        }
        // observation is a full vector of count i.e., the value at v-th element is
        // the number of times v-th word were observed
        else if (observation instanceof int[])
        {
            int[] x = (int[])observation;
            if (x.length != this.dim)
                throw new Exception("Size of count vector does not match");

            long sumMM = 0;
            double lgsum1 = 0; // \sum_{i=1}^V (\psi(x[i])-\psi(\sum_{i=1}^V x[i]))*x[i]
            double lgsum2 = 0; //  {\log\Gamma(\sum_{i=1}^V x[i]+1) -\sum_{i=1}^V  \log\Gamma( x[i]+1)}

            for (int i = 0; i < x.length; i++)
            {
                sumMM += x[i];
                lgsum1 += (Gamma.digamma(postParam[i]) - Gamma.digamma(postParamSum))*x[i];
                lgsum2 -= Gamma.logGamma(1 + x[i]);
            }
            lgsum2 += Gamma.logGamma(sumMM + 1);

            ll = lgsum1 + lgsum2;

        }
        else if (observation instanceof SparseVector)
        {
            SparseVector x = (SparseVector)observation;
            if (x.getNonzeoLength() == 0)
                return 0;

            if (x.getLength() != this.dim)
                throw new Exception("Size of count vector does not match");

            long sumMM = 0;
            double lgsum1 = 0; // \sum_{i=1}^V (\psi(x[i])-\psi(\sum_{i=1}^V x[i]))*x[i]
            double lgsum2 = 0; //  {\log\Gamma(\sum_{i=1}^V x[i]+1) -\sum_{i=1}^V  \log\Gamma( x[i]+1)}

            Integer[] index=x.getIndices();

            for (int i = 0; i < index.length; i++) {
                int value=(int)x.getValue(index[i]);
                sumMM += value;
                lgsum1 += (Gamma.digamma(postParam[index[i]]) - Gamma.digamma(postParamSum))*value;
                lgsum2 -= Gamma.logGamma(1 + value);
            }
            lgsum2 += Gamma.logGamma(sumMM + 1);

            ll = lgsum1 + lgsum2;

        }
        else
        throw new Exception("Data item type is not supported");

        return ll;
    }

    public void stochasticUpdate(BayesianComponent pp, double kappa) throws  Exception{
        if (!pp.getClass().getName().contains("MultinomialDirichlet"))
            throw  new Exception("Data item type is not supported");

//        double[] ppParams = ((MultinomialDirichlet)pp).getPosteriorParameters();
        //double[] ppParams = ((MultinomialDirichlet)pp).getSufficientStatistics();

        for (int i = 0; i < this.dim; i++)
            this.mi.setValue(i,(1 - kappa) * this.mi.getValue(i) + kappa * ((MultinomialDirichlet) pp).mi.getValue(i));
        this.mm=(1 - kappa)*this.mm+kappa*((MultinomialDirichlet) pp).mm;
    }
    public double[] getPosteriorParameters()
    {
        double[] post = new double[this.dim];
        for (int i = 0; i < this.dim; i++)
            post[i] = this.aa + this.mi.getValue(i);
        return post;
    }
    public SparseVector getSufficientStatistics()
    {
//        double[] post = new double[this.dim];
//        for (int i = 0; i < this.dim; i++)
//            post[i] = this.mi.getValue(i);
//        return post;
        return this.mi;
    }
    public void plus(BayesianComponent pp) throws  Exception{

        if (!pp.getClass().getName().contains("MultinomialDirichlet"))
            throw  new Exception("Data item type is not supported");

        if ( this.dim!=((MultinomialDirichlet)pp).dim)
            throw new Exception("Size of distribution does not match");

        this.numItems += ((MultinomialDirichlet)pp).numItems;
        this.mm += ((MultinomialDirichlet)pp).mm;
        this.mi.plus(((MultinomialDirichlet)pp).mi);
//        for (int i = 0; i < this.dim; i++)
//        {
//            this.mi.setValue(i, this.mi.getValue(i)+ ((MultinomialDirichlet)pp).mi.getValue(i));
//        }

    }

    public Object clone(){
        MultinomialDirichlet out= new MultinomialDirichlet(dim,aa*dim);
        out.mi = this.mi.clone(0,dim);
        out.mm = this.mm;
        out.numItems=this.numItems;
        out.Z0=this.Z0 ;
        return out;
    }

    public static double[][] expectationLogLikelihood(ArrayList<Integer> observations, ArrayList<BayesianComponent> pp)
    {
        int nCount = observations.size();
        int nDists = pp.size();
        double[] sumPostparamsDigamma = new double[nDists];
        double[][] postParams = new double[nDists][];
        double[][] Ell = new double[nCount][];

        for (int kk = 0; kk < nDists; kk++)
        {
            postParams[kk] = ((MultinomialDirichlet)pp.get(kk)).getPosteriorParameters();
            for (int dd = 0; dd < ((MultinomialDirichlet)pp.get(kk)).dim; dd++)
            {
                sumPostparamsDigamma[kk] += postParams[kk][dd];
            }
            sumPostparamsDigamma[kk] = Gamma.digamma(sumPostparamsDigamma[kk]);
        }

        for (int ii = 0; ii < nCount; ii++)
        {
            Ell[ii] = new double[nDists];
            for (int kk = 0; kk < nDists; kk++)
            {
                Ell[ii][kk] = Gamma.digamma(postParams[kk][(int)observations.get(ii)]) - sumPostparamsDigamma[kk];
            }
        }

        return Ell;
    }
    // This for compute expectationLogLikelihood of multiple obs w.r.t multiple atoms consider data type
    public static double[][] expectationLogLikelihoodGeneric(ArrayList<Object> observations, ArrayList<BayesianComponent> pp)  throws Exception {
        int nCount = observations.size();
        int nDists = pp.size();
        double[] sumPostparamsDigamma = new double[nDists];
        double[][] postParams = new double[nDists][];
        double[][] Ell = new double[nCount][];
        if (observations.get(0) instanceof SparseVector) {

            for (int kk = 0; kk < nDists; kk++)
            {
                postParams[kk] = ((MultinomialDirichlet)pp.get(kk)).getPosteriorParameters();
                for (int dd = 0; dd < ((MultinomialDirichlet)pp.get(kk)).dim; dd++)
                {
                    sumPostparamsDigamma[kk] += postParams[kk][dd];
                }
                sumPostparamsDigamma[kk] = Gamma.digamma(sumPostparamsDigamma[kk]);
            }

            for (int ii = 0; ii < nCount; ii++)
            {
                SparseVector x = (SparseVector) observations.get(ii);
                Ell[ii] = new double[nDists];
                for (int kk = 0; kk < nDists; kk++)
                {
                    if (x.getNonzeoLength() == 0){
                        Ell[ii][kk] = 0;
                        continue;
                    }

                    long sumMM = 0;
                    double lgsum1 = 0; // \sum_{i=1}^V (\psi(x[i])-\psi(\sum_{i=1}^V x[i]))*x[i]
                    double lgsum2 = 0; //  {\log\Gamma(\sum_{i=1}^V x[i]+1) -\sum_{i=1}^V  \log\Gamma( x[i]+1)}

                    Integer[] index = x.getIndices();

                    for (int i = 0; i < index.length; i++) {
                        int value = (int) x.getValue(index[i]);
                        sumMM += value;
                        lgsum1 += (Gamma.digamma(postParams[kk][index[i]]) - Gamma.digamma(sumPostparamsDigamma[kk])) * value;
                        lgsum2 -= Gamma.logGamma(1 + value);
                    }
                    lgsum2 += Gamma.logGamma(sumMM + 1);

                    Ell[ii][kk] = lgsum1 + lgsum2;
                }
            }
            return Ell;
        } else
            throw new Exception("Data item type is not supported");
    }

    @Override
    public String toString() {

        return "MultinomialDirichlet\n\tPrior: \t\t dim="+dim+", alpha="+aa+
                "\n\tPosterior:   numItems="+numItems+", pseudoCount= ["+mi+"]\n";
    }
}
