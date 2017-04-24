package org.bnpstat.stats.conjugacy;

/**
 * Created by hvhuynh on 2/13/2017.
 */
public class ProductSpace implements BayesianComponent  {
    private double numItems;
    private int numSpaces;
    private BayesianComponent[] spaces;

    public ProductSpace(BayesianComponent[] spaces ){
        this.spaces=spaces;
        numSpaces=spaces.length;
        numItems=0;
    }

    public double getNumItems() {
        return numItems;
    }

    public void setNumItems(double numItems) {
        this.numItems = numItems;
    }

    public int getNumSpaces() {
        return numSpaces;
    }

    @Override
    public void add(Object observation) throws Exception {
        if (ProductSpaceData.class.isInstance(observation)) {
            ProductSpaceData obs=(ProductSpaceData)observation;
            numItems++;
            for(int i=0;i<numSpaces;i++){
                spaces[i].add(obs.getSpace(i));
            }
        }else
            throw new Exception("Data item type is not supported. Used array of objects instead");
    }

    @Override
    public void add(Object observation, double prob) throws Exception {
        if (ProductSpaceData.class.isInstance(observation)) {
            ProductSpaceData obs=(ProductSpaceData)observation;
            numItems++;
            for(int i=0;i<numSpaces;i++){
                spaces[i].add(obs.getSpace(i),prob);
            }
        }else
            throw new Exception("Data item type is not supported. Used array of objects instead");
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
        else {
            if (ProductSpaceData.class.isInstance(observation)) {
                ProductSpaceData obs = (ProductSpaceData) observation;
                for (int i = 0; i < numSpaces; i++) {
                    spaces[i].remove(obs.getSpace(i));
                }
            }else
                throw new Exception("Data item type is not supported. Used array of objects instead");
        }
    }

    @Override
    public Object getPosterior() throws Exception {
        throw new Exception("The method is not implemented yet!");
    }

    @Override
    public double logPredictive(Object observation) throws Exception {
        double val=0;
        if (ProductSpaceData.class.isInstance(observation)) {
            ProductSpaceData obs = (ProductSpaceData) observation;
            for (int i = 0; i < numSpaces; i++) {
                val+=spaces[i].logPredictive(obs.getSpace(i));
            }
        }else
            throw new Exception("Data item type is not supported. Used array of objects instead");
        return val;
    }

    @Override
    public double logMarginal() throws Exception {
        throw new Exception("The method is not implemented yet!");
    }

    @Override
    public double expectationLogLikelihood(Object observation) throws Exception {
        double val=0;
        if (ProductSpaceData.class.isInstance(observation)) {
            ProductSpaceData obs = (ProductSpaceData) observation;
            for (int i = 0; i < numSpaces; i++) {
                val+=spaces[i].expectationLogLikelihood(obs.getSpace(i));
            }
        }else
            throw new Exception("Data item type is not supported. Used array of objects instead");
        return val;
    }

    @Override
    public void stochasticUpdate(BayesianComponent pp, double kappa) throws Exception {
        if (ProductSpace.class.isInstance(pp)) {
            for (int i = 0; i < numSpaces; i++) {
                spaces[i].stochasticUpdate(((ProductSpace) pp).spaces[i],kappa);
            }
        }else
            throw new Exception("BayesianComponent is not a ProductSpace object!");
    }

    @Override
    public void plus(BayesianComponent pp) throws Exception {
        if (ProductSpace.class.isInstance(pp)) {
            for (int i = 0; i < numSpaces; i++) {
                spaces[i].plus(((ProductSpace) pp).spaces[i]);
            }
        }else
            throw new Exception("BayesianComponent is not a ProductSpace object!");
    }
    public Object clone() {
        BayesianComponent[] components = new BayesianComponent[numSpaces];
        for (int i = 0; i < numSpaces; i++) {
            components[i] = (BayesianComponent) spaces[i].clone();
        }
        return new ProductSpace(components);
    }

}
