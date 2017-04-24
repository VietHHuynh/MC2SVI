package org.bnpstat.util;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import org.apache.jute.Index;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

/**
 * Created by hvhuynh on 4/30/2016.
 */
public  class FastMatrixFunctions {
    /**
     * This function allow to compute product of two matrices A(KKxTTxMM) with B(NNxMM) resulting out C (NNxKKxTT) where
     *  C(nn,kk,tt)=\sum_mm(A(kk,tt,mm)*B(nn,mm)). This implementation uses JBLAS library.
     * @param mat1 - A matrix (KKxTTxMM)
     * @param mat2 - B matrix (NNxMM)
     * @return  C(NNxKKxTT)
     */
    public static double[][][] multNDArrayJBLAS(double[][][] mat1, double[][] mat2) {

        int KK=mat1.length;
        int TT=mat1[0].length;
        int MM=mat1[0][0].length;
        int NN= mat2.length;
        DoubleMatrix temp1 = new DoubleMatrix(KK*TT,MM);
        IntervalRange colRange=new IntervalRange(0,MM);
        for(int kk=0;kk<KK;kk++){
            IntervalRange rowRange=new IntervalRange(TT*kk,TT*(kk+1));
            temp1.put(rowRange,colRange, new DoubleMatrix(mat1[kk]));
        }
        DoubleMatrix temp = new DoubleMatrix(mat2);

        DoubleMatrix temp2=temp.mmul(temp1.transpose());
        double[][][] result=new double[NN][KK][TT];
        colRange=new IntervalRange(0,KK*TT);
        for(int nn=0;nn<NN;nn++){
            IntervalRange rowRange=new IntervalRange(nn,nn+1);
            result[nn]=temp2.get(rowRange,colRange).reshape(TT,KK).transpose().toArray2();
        }
        return result;
    }
    /**
     * This function allow to compute the product of two matrices A(KKxTTxMM) with B(NNxMM)  and add matrix C(KKxTT) resulting out D (NNxKKxTT) where
     *  D(nn,kk,tt)=\sum_mm(A(kk,tt,mm)*B(nn,mm))+c(kk,tt). This implementation uses JBLAS library.
     * @param mat1 - A matrix (KKxTTxMM)
     * @param mat2 - B matrix (NNxMM)
     * @param mat3 - C matrix (KKxTT)
     * @return  D(NNxKKxTT)
     */
    public static double[][][] multNDArrayJBLAS(double[][][] mat1, double[][] mat2,double[][]mat3) {

        int KK=mat1.length;
        int TT=mat1[0].length;
        int MM=mat1[0][0].length;
        int NN= mat2.length;

        DoubleMatrix temp1 = new DoubleMatrix(KK*TT,MM);
        IntervalRange colRange=new IntervalRange(0,MM);
        for(int kk=0;kk<KK;kk++){
            IntervalRange rowRange=new IntervalRange(TT*kk,TT*(kk+1));
            temp1.put(rowRange,colRange, new DoubleMatrix(mat1[kk]));
        }
        DoubleMatrix temp = new DoubleMatrix(mat2);

        DoubleMatrix temp2=temp.mmul(temp1.transpose());
        DoubleMatrix temp3=new DoubleMatrix(mat3);
        double[][][] result=new double[NN][KK][TT];
        colRange=new IntervalRange(0,KK*TT);
        for(int nn=0;nn<NN;nn++){
            IntervalRange rowRange=new IntervalRange(nn,nn+1);
            result[nn]=temp2.get(rowRange,colRange).reshape(TT,KK).transpose().addi(temp3).toArray2();
        }
        return result;
    }



    /**
     * This function allow to compute the product of two matrices A(NNxKKxTT) with B(KKxTTxMM)  and matrix C(KK) resulting out D (NNxMM) where
     *  D(nn,mm)=\sum_{kk,tt}(A(nn,kk,tt)*B(kk,tt,mm)*C(kk)). This implementation uses JBLAS library.
     * @param mat1 - A matrix (NNxKKxTT)
     * @param mat2 - B matrix (KKxTTxMM)
     * @param mat3 - C matrix (KK)
     * @return  D(NNxMM)
     */

    public static double[][] multNDArrayJBLAS(double[][][] mat1, double[][][] mat2,double[]mat3) {

        int NN=mat1.length;
        int KK= mat2.length;
        int TT=mat2[0].length;
        int MM=mat2[0][0].length;

        DoubleMatrix temp1 = new DoubleMatrix(NN,KK*TT);
        for(int nn=0;nn<NN;nn++){
            temp1.putRow(nn, new DoubleMatrix(mat1[nn]).transpose());
        }
        DoubleMatrix temp2=(new DoubleMatrix(mat3)).repmat(1,TT).transpose().reshape(1,KK*TT);
        temp1.muliRowVector(temp2);

        DoubleMatrix temp3 = new DoubleMatrix(KK*TT,MM);
        IntervalRange colRange=new IntervalRange(0,MM);
        for(int kk=0;kk<KK;kk++){
            IntervalRange  rowRange=new IntervalRange(TT*kk,TT*(kk+1));
            temp3.put(rowRange,colRange, new DoubleMatrix(mat2[kk]));
        }
        return temp1.mmul(temp3).toArray2();
    }

}
