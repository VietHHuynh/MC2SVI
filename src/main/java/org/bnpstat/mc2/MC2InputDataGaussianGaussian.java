package org.bnpstat.mc2;

import org.bnpstat.maths.SparseVector;

import java.util.ArrayList;

/**
 * Created by hvhuynh on 2/17/2017.
 */
public class MC2InputDataGaussianGaussian {
    public int contentDim;
    public int contextDim;
    public int ngroups;
    public int[] numData;
    public ArrayList<double[]> xx; // context Data
    public ArrayList<double[]>[] ss; // content data
}
