package org.bnpstat.stats.conjugacy;

import java.io.Serializable;

/**
 * Created by hvhuynh on 1/29/2017.
 */
public  class DoubleMultivariateData implements Serializable {
    private  double[] observation;
    public DoubleMultivariateData(double[] obs){
        observation=obs;
    }
    public double get(int index){
        return observation[index];
    }
    public double[] getVal(){
        return observation;
    }
}