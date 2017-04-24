package org.bnpstat.maths;

import java.util.HashMap;
import java.util.Set;

/**
 * Created by hvhuynh on 11/6/2015.
 */
public class SparseVector extends AbstractVector {
    HashMap<Integer, Double> values;

    public SparseVector(int length) {
        values = new HashMap<>();
        dim = length;
    }

    public SparseVector(double[] inValues) {
        values = new HashMap<>();
        dim = inValues.length;
        for (int i = 0; i < inValues.length; ++i) {
            if (inValues[i] > 0)
                values.put(i, inValues[i]);
        }
    }

    public SparseVector(int[] indices, double[] inValues, int dim) {
        values = new HashMap<>();
        this.dim = dim;
        for (int i = 0; i < indices.length; ++i) {
            values.put(indices[i], inValues[i]);
        }
    }

    public SparseVector(SparseVector copy) {
        dim = copy.dim;
        values = (HashMap<Integer, Double>) values.clone();
    }

    public Integer[] getIndices() {
        return values.keySet().toArray(new Integer[values.size()]);
    }

    public int getLength() {
        return dim;
    }

    public int getNonzeoLength() {
        return values.size();
    }

    public SparseVector clone() {
        SparseVector nVector = new SparseVector(dim);
        Object[] indices = this.values.keySet().toArray();
        for (int i = 0; i < indices.length; i++) {
            nVector.setValue((Integer) indices[i], this.values.get(indices[i]));
        }
        return nVector;
    }

    public SparseVector clone(int index, int count) {
        SparseVector nVector = new SparseVector(count);
        Object[] indices = this.values.keySet().toArray();
        for (int i = 0; i < indices.length; i++) {
            Integer key = (Integer) indices[i] - index;
            if (key > 0 && key < count)
                nVector.setValue(key, this.values.get(indices[i]));
        }
        return nVector;
    }

    public double getValue(int index) {

        return values.containsKey(index) ? values.get(index) : 0;
    }

    public void setValue(int index, double value) {
        if (value != 0) values.put(index, value);
    }

    public void addValue(int index, double value) {
        values.put(index, getValue(index) + value);
    }

    public void subValue(int index, double value) {
        values.put(index, getValue(index) - value);
    }

    public SparseVector cloneVector(int index, int count) {
        return clone(index, count);
    }

    public void plus(SparseVector v) {
        Integer[] vIndices = v.getIndices();
        for (int ii = 0; ii < vIndices.length; ii++)
            this.values.put(vIndices[ii], this.getValue(vIndices[ii]) + v.getValue(vIndices[ii]));

    }

    public void sub(SparseVector v) {
        Integer[] vIndices = v.getIndices();
        for (int ii = 0; ii < vIndices.length; ii++)
            this.values.put(vIndices[ii], this.getValue(vIndices[ii]) - v.getValue(vIndices[ii]));

    }

    public void mul(SparseVector v) {
        HashMap<Integer, Double> temp = (HashMap) this.values.clone();
        temp.putAll(v.values);
        Integer[] indices = temp.keySet().toArray(new Integer[values.size()]);
        for (int ii = 0; ii < indices.length; ii++) {
            if (this.getValue(indices[ii]) == 0 || v.getValue(indices[ii]) == 0)
                this.values.remove(indices[ii]);
            else
                this.values.put(indices[ii], this.getValue(indices[ii]) * v.getValue(indices[ii]));
        }

    }

    public void mul(double val) {

        Integer[] indices = this.getIndices();
        for (int ii = 0; ii < indices.length; ii++)
            this.values.put(indices[ii], this.getValue(indices[ii]) * val);

    }

    public void plus(double val) {
        Integer[] indices = this.getIndices();
        for (int ii = 0; ii < indices.length; ii++)
            this.values.put(indices[ii], this.getValue(indices[ii]) + val);
    }

    public void sub(double val) {
        plus(-val);
    }

    public void div(double val) throws Exception {
        if (val != 0)
            mul(1 / val);
        else
            throw new Exception("Divide by zero");
    }

    public AbstractVector createVector(int length) {
        return new SparseVector(length);
    }

    @Override
    public String toString() {
        Integer[] indices = getIndices();
        if (indices.length == 0) return "Empty";
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < indices.length; i++)
            stringBuilder.append(indices[i] + ":" + getValue(indices[i]) + "  ");
        return stringBuilder.substring(0, stringBuilder.length() - 2);
    }
}
