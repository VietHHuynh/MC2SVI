package org.bnpstat.mc2;

import org.bnpstat.stats.conjugacy.BayesianComponent;

import java.io.Serializable;

/**
 * Created by hvhuynh on 11/27/2015.
 */
public class MC2Parameters implements Serializable {
    public int KK;
    public int TT;
    public int MM;
    public double aa;
    public double ee;
    public double gg;
    public BayesianComponent q0;
    public BayesianComponent p0;
    public int ngroups;
    public double[][][] qcc;

    public MC2Parameters(int KK, int TT, int MM, double aa, double gg, double ee, int ngroups, double[][][] qcc, BayesianComponent q0, BayesianComponent p0) {
        this.KK = KK;
        this.TT = TT;
        this.MM = MM;
        this.aa = aa;
        this.ee = ee;
        this.gg = gg;
        this.ngroups = ngroups;
        this.q0 = q0;
        this.p0 = p0;
        this.qcc = qcc;

    }
}
