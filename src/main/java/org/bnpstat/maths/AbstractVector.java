package org.bnpstat.maths;

import java.io.Serializable;

/**
 * Created by hvhuynh on 11/6/2015.
 */
public abstract class AbstractVector implements Serializable {
    public int dim;

    public abstract int getLength();

    public abstract double getValue(int index);

    public abstract void setValue(int index, double value);

    public abstract AbstractVector cloneVector(int index, int count);

    public abstract AbstractVector createVector(int length);

}
