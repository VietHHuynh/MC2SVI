package org.bnpstat.mc2;

import org.bnpstat.maths.SparseVector;

import java.io.Serializable;

/**
 * Created by hvhuynh on 19/11/2015.
 */

/**
 * This class represents input data for MC2 model with Multinomial context and Categorical content
 */
public class MC2InputDataMultCat implements Serializable {
    public int contentDim;
    public int contextDim;
    public int ngroups;
    public int[] numData;
    public SparseVector[] xx; //context data
    public int[][] ss; // content data
}
