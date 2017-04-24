package org.bnpstat.mc2;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by hvhuynh on 17/11/2015.
 */
public class MC2Data implements Serializable {
    public int id;
    public Object xx;
    public ArrayList<Object> ss;

    public MC2Data(int inId, Object inXX, ArrayList<Object> inSS) {
        id = inId;
        xx = inXX;
        ss = inSS;
    }
}