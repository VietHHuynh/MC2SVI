package org.bnpstat.stats.conjugacy;

import java.util.Objects;

/**
 * Created by hvhuynh on 2/13/2017.
 */
public class ProductSpaceData {
    private Object[] data;
    private int numSpaces;
    public ProductSpaceData(Object[] dataPoint){
        data=dataPoint;
        numSpaces=dataPoint.length;
    }
    public Object getSpace(int index){
        if (0<=index&&index<numSpaces )
            return data[index];
        else
            throw new ArrayIndexOutOfBoundsException("The index is less than 0 or greater than the number of spaces: "+numSpaces);
    }

}
