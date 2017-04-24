package org.bnpstat.mc2;

import org.bnpstat.stats.conjugacy.BayesianComponent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by hvhuynh on 11/11/2015.
 */
public class MC2MapReduceOutput implements Serializable {
    public double[][] docRhoshat;
    public double[][][] docChishat;
    public double[][][] docZetashat;
    public double[][] docVarphishat;

    public ArrayList<BayesianComponent> docAlphahat;
    public ArrayList<BayesianComponent> docLambdahat;
    // This is new variables for computing clustering performance
    public HashMap<Integer, double[]> qzz;
}
