package org.bnpstat.util;

import com.jmatio.io.MatFileWriter;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.*;
import org.bnpstat.maths.SparseVector;
import org.bnpstat.mc2.MC2InputDataMultCat;
import org.bnpstat.stats.conjugacy.BayesianComponent;
import org.bnpstat.stats.conjugacy.MultinomialDirichlet;
import org.bnpstat.stats.conjugacy.MultivariateGaussianGamma;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by hvhuynh on 11/10/2015.
 */
public class MatlabJavaConverter {
    public static void exportMC2SVIResultToMat(int KK, int TT, int MM, double[][][] qcc, double[][] rhos, double[][][] zetas,
                                               double[][] varphis, int ngroups, int[] numdata, double elapse,
                                               ArrayList<BayesianComponent> qq, ArrayList<BayesianComponent> pp,
                                               String strOut, int IsPlusOne) throws  Exception
    {
        //#region Convert data to Matlab objects
        //KK
        short[] tempKK = new short[1];
        tempKK[0] =(short)KK;
        MLInt16 KKint = new MLInt16("KK", tempKK, 1);

        //TT
        short[] tempTT = new short[1];
        tempTT[0] = (short)TT;
        MLInt16 TTint = new MLInt16("TT", tempTT, 1);

        //MM
        short[] tempMM = new short[1];
        tempMM[0] = (short)MM;
        MLInt16 MMint = new MLInt16("MM", tempMM, 1);

        //ngroups
        short[] tempngroups = new short[1];
        tempngroups[0] = (short)ngroups;
        MLInt16 ngroupsint = new MLInt16("ngroups", tempngroups, 1);

        //numdata
        short[] sNumdata=new short[numdata.length];
        for (int i=0;i<sNumdata.length;i++){
            sNumdata[i]=(short)numdata[i];
        }
        MLInt16 numdataint = new MLInt16("numdata", sNumdata, 1);

        //elapse time
        double[] tempelapse = new double[1];
        tempelapse[0] = elapse;
        MLDouble elapsedouble = new MLDouble("elapse", tempelapse, 1);

        ////qzz
        //MLDouble qzzdouble = new MLDouble("qzz", qzz, 1);

        //qcc
        int [] dims= new int[2];
        dims[0]=qcc.length;
        dims[1]=1;
        MLCell qcccell = new MLCell("qcc", dims);
        for (int kk = 0; kk < qcc.length; kk++)
        {
            double[][] temp = new double[qcc[0].length][];
            for (int tt = 0; tt < qcc[0].length; tt++) {
                temp[tt] = new double[qcc[0][0].length];
                for (int mm = 0; mm < qcc[0][0].length; mm++)
                    temp[tt][mm] = qcc[kk][tt][mm];
            }

            MLDouble celltt = new MLDouble("temp", temp);
            qcccell.set( celltt,kk);

        }

        //rhos
        double[][] temprhos = new double[rhos.length][];
        for (int kk = 0; kk < rhos.length; kk++)
        {
            temprhos[kk] = new double[2];
            temprhos[kk][0] = rhos[kk][0];
            temprhos[kk][1] = rhos[kk][1];
        }
        MLDouble rhosdouble = new MLDouble("rhos", temprhos);

        //varphis
        double[][] tempvarphis = new double[varphis.length][];
        for (int mm = 0; mm < varphis.length; mm++)
        {
            tempvarphis[mm] = new double[2];
            tempvarphis[mm][0] = varphis[mm][0];
            tempvarphis[mm][1] = varphis[mm][1];
        }
        MLDouble varphisdouble = new MLDouble("varphis", tempvarphis);


        //zetas
        dims= new int[2];
        dims[0]=zetas.length;
        dims[1]=1;
        MLCell zetascell = new MLCell("zetas", dims);
        for (int kk = 0; kk < zetas.length; kk++)
        {
            double[][] temp = new double[zetas[0].length][];
            for (int tt = 0; tt < zetas[0].length; tt++)
            {
                temp[tt] = new double[2];
                temp[tt][0] = zetas[kk][tt][0];
                temp[tt][1] = zetas[kk][tt][1];
            }

            MLDouble celltt = new MLDouble("temp", temp);
            zetascell.set(celltt,kk);

        }




        //qq
        double[][] tempQQ = new double[qq.size()][];
        if (qq.get(0).getClass().getName().contains("MultinomialDirichlet"))
        {
            for (int kk = 0; kk < qq.size(); kk++)
            {
                tempQQ[kk] = ((MultinomialDirichlet)qq.get(kk)).getPosteriorParameters();
            }
        }else  if (qq.get(0).getClass().getName().contains("MultivariateGaussianGamma")){
            for (int kk = 0; kk < qq.size(); kk++)
            {
                double[] temp= ((MultivariateGaussianGamma)qq.get(kk)).getPostMean();
                tempQQ[kk] = new double[temp.length+3];
                System.arraycopy(temp,0,tempQQ[kk],0,temp.length);
                tempQQ[kk][temp.length]=((MultivariateGaussianGamma)qq.get(kk)).getAn();
                tempQQ[kk][temp.length+1]=((MultivariateGaussianGamma)qq.get(kk)).getBn();
                tempQQ[kk][temp.length+2]=((MultivariateGaussianGamma)qq.get(kk)).getSn();
            }
        }
        else
        {
            throw new Exception("Distribution currently is not supported for saving Matlab file!");
        }
        MLDouble qqDouble = new MLDouble("qq", tempQQ);

        //pp
        double[][] tempPP = new double[pp.size()][];
        if (pp.get(0).getClass().getName().contains("MultinomialDirichlet"))
        {
            for (int kk = 0; kk < pp.size(); kk++)
            {
                tempPP[kk] = ((MultinomialDirichlet)pp.get(kk)).getPosteriorParameters();
            }
        } else  if (pp.get(0).getClass().getName().contains("MultivariateGaussianGamma")){
        for (int kk = 0; kk < pp.size(); kk++)
        {
            double[] temp= ((MultivariateGaussianGamma)pp.get(kk)).getPostMean();
            tempPP[kk] = new double[temp.length+3];
            System.arraycopy(temp,0,tempPP[kk],0,temp.length);
            tempPP[kk][temp.length]=((MultivariateGaussianGamma)pp.get(kk)).getAn();
            tempPP[kk][temp.length+1]=((MultivariateGaussianGamma)pp.get(kk)).getBn();
            tempPP[kk][temp.length+2]=((MultivariateGaussianGamma)pp.get(kk)).getSn();
        }
    }
        else
        {
            throw new Exception("Distribution currently is not supported for saving Matlab file!");
        }
        MLDouble ppDouble = new MLDouble("pp", tempPP);


        //#region write arrays to file

        ArrayList<MLArray> list = new ArrayList<MLArray>();
        list.add(KKint);
        list.add(TTint);
        list.add(MMint);
        list.add(numdataint);
        list.add(ngroupsint);
        list.add(qcccell);
        list.add(rhosdouble);
        list.add(zetascell);
        list.add(varphisdouble);
        list.add(qqDouble);
        list.add(ppDouble);
        list.add(elapsedouble);
        MatFileWriter writer = new MatFileWriter(strOut, list);


    }
    public static void exportMC2SVIResultToMatWithQzz(int KK, int TT, int MM, double[][][] qcc, double[][] rhos, double[][][] zetas,
                                                      double[][] varphis, HashMap<Integer,double[]> qzz, int ngroups, int[] numdata, double elapse,
                                                      ArrayList<BayesianComponent> qq, ArrayList<BayesianComponent> pp,
                                                      String strOut, int IsPlusOne) throws  Exception
    {
        //#region Convert data to Matlab objects
        //KK
        short[] tempKK = new short[1];
        tempKK[0] =(short)KK;
        MLInt16 KKint = new MLInt16("KK", tempKK, 1);

        //TT
        short[] tempTT = new short[1];
        tempTT[0] = (short)TT;
        MLInt16 TTint = new MLInt16("TT", tempTT, 1);

        //MM
        short[] tempMM = new short[1];
        tempMM[0] = (short)MM;
        MLInt16 MMint = new MLInt16("MM", tempMM, 1);

        //ngroups
        short[] tempngroups = new short[1];
        tempngroups[0] = (short)ngroups;
        MLInt16 ngroupsint = new MLInt16("ngroups", tempngroups, 1);

        //numdata
        short[] sNumdata=new short[numdata.length];
        for (int i=0;i<sNumdata.length;i++){
            sNumdata[i]=(short)numdata[i];
        }
        MLInt16 numdataint = new MLInt16("numdata", sNumdata, 1);

        //elapse time
        double[] tempelapse = new double[1];
        tempelapse[0] = elapse;
        MLDouble elapsedouble = new MLDouble("elapse", tempelapse, 1);

        ////qzz
        //MLDouble qzzdouble = new MLDouble("qzz", qzz, 1);

        //qcc
        int [] dims= new int[2];
        dims[0]=qcc.length;
        dims[1]=1;
        MLCell qcccell = new MLCell("qcc", dims);
        for (int kk = 0; kk < qcc.length; kk++)
        {
            double[][] temp = new double[qcc[0].length][];
            for (int tt = 0; tt < qcc[0].length; tt++) {
                temp[tt] = new double[qcc[0][0].length];
                for (int mm = 0; mm < qcc[0][0].length; mm++)
                    temp[tt][mm] = qcc[kk][tt][mm];
            }

            MLDouble celltt = new MLDouble("temp", temp);
            qcccell.set( celltt,kk);

        }

        //rhos
        double[][] temprhos = new double[rhos.length][];
        for (int kk = 0; kk < rhos.length; kk++)
        {
            temprhos[kk] = new double[2];
            temprhos[kk][0] = rhos[kk][0];
            temprhos[kk][1] = rhos[kk][1];
        }
        MLDouble rhosdouble = new MLDouble("rhos", temprhos);

        //varphis
        double[][] tempvarphis = new double[varphis.length][];
        for (int mm = 0; mm < varphis.length; mm++)
        {
            tempvarphis[mm] = new double[2];
            tempvarphis[mm][0] = varphis[mm][0];
            tempvarphis[mm][1] = varphis[mm][1];
        }
        MLDouble varphisdouble = new MLDouble("varphis", tempvarphis);


        //zetas
        dims= new int[2];
        dims[0]=zetas.length;
        dims[1]=1;
        MLCell zetascell = new MLCell("zetas", dims);
        for (int kk = 0; kk < zetas.length; kk++)
        {
            double[][] temp = new double[zetas[0].length][];
            for (int tt = 0; tt < zetas[0].length; tt++)
            {
                temp[tt] = new double[2];
                temp[tt][0] = zetas[kk][tt][0];
                temp[tt][1] = zetas[kk][tt][1];
            }

            MLDouble celltt = new MLDouble("temp", temp);
            zetascell.set(celltt,kk);

        }

        //qzz
        Object[] docIDs=qzz.keySet().toArray();

        double[][] tempqzz= new double[docIDs.length][];
        for (int ii=0;ii<docIDs.length;ii++){
            double[]temp=qzz.get(docIDs[ii]);
            tempqzz[ii]= new double[temp.length+1];
            tempqzz[ii][0]=(Integer)docIDs[ii];
            System.arraycopy(temp,0,tempqzz[ii],1,temp.length);

        }

        MLDouble qzzdouble = new MLDouble("qzzs", tempqzz);

        //qq
        double[][] tempQQ = new double[qq.size()][];
        if (qq.get(0).getClass().getName().contains("MultinomialDirichlet"))
        {
            for (int kk = 0; kk < qq.size(); kk++)
            {
                tempQQ[kk] = ((MultinomialDirichlet)qq.get(kk)).getPosteriorParameters();
            }
        }else  if (qq.get(0).getClass().getName().contains("MultivariateGaussianGamma")){
            for (int kk = 0; kk < qq.size(); kk++)
            {
                double[] temp= ((MultivariateGaussianGamma)qq.get(kk)).getPostMean();
                tempQQ[kk] = new double[temp.length+3];
                System.arraycopy(temp,0,tempQQ[kk],0,temp.length);
                tempQQ[kk][temp.length]=((MultivariateGaussianGamma)qq.get(kk)).getAn();
                tempQQ[kk][temp.length+1]=((MultivariateGaussianGamma)qq.get(kk)).getBn();
                tempQQ[kk][temp.length+2]=((MultivariateGaussianGamma)qq.get(kk)).getSn();
            }
        }
        else
        {
            throw new Exception("Distribution currently is not supported for saving Matlab file!");
        }
        MLDouble qqDouble = new MLDouble("qq", tempQQ);

        //pp
        double[][] tempPP = new double[pp.size()][];
        if (pp.get(0).getClass().getName().contains("MultinomialDirichlet"))
        {
            for (int kk = 0; kk < pp.size(); kk++)
            {
                tempPP[kk] = ((MultinomialDirichlet)pp.get(kk)).getPosteriorParameters();
            }
        }else  if (pp.get(0).getClass().getName().contains("MultivariateGaussianGamma")){
            for (int kk = 0; kk < pp.size(); kk++)
            {
                double[] temp= ((MultivariateGaussianGamma)pp.get(kk)).getPostMean();
                tempPP[kk] = new double[temp.length+3];
                System.arraycopy(temp,0,tempPP[kk],0,temp.length);
                tempPP[kk][temp.length]=((MultivariateGaussianGamma)pp.get(kk)).getAn();
                tempPP[kk][temp.length+1]=((MultivariateGaussianGamma)pp.get(kk)).getBn();
                tempPP[kk][temp.length+2]=((MultivariateGaussianGamma)pp.get(kk)).getSn();
            }
        }
        else
        {
            throw new Exception("Distribution currently is not supported for saving Matlab file!");
        }
        MLDouble ppDouble = new MLDouble("pp", tempPP);


        //#region write arrays to file

        ArrayList<MLArray> list = new ArrayList<MLArray>();
        list.add(KKint);
        list.add(TTint);
        list.add(MMint);
        list.add(numdataint);
        list.add(ngroupsint);
        list.add(qcccell);
        list.add(rhosdouble);
        list.add(zetascell);
        list.add(varphisdouble);
        list.add(qqDouble);
        list.add(ppDouble);
        list.add(elapsedouble);
        list.add(qzzdouble);
        MatFileWriter writer = new MatFileWriter(strOut, list);


    }
    public static MC2InputDataMultCat readMC2DataFromMatFiles(String contVobSizeHeader, String cxtVobSizeHeader,  String strMat)
    {
        System.out.println("Start loading Mat files...");
        MC2InputDataMultCat outData=null;
        try {
             outData= new MC2InputDataMultCat();
            MatFileReader mr = new MatFileReader(strMat);
            //Content Vocabulary size
            outData.contentDim = (int)readScalarVariable(mr, contVobSizeHeader);
            //Context Vocabulary Size
            outData.contextDim = (int)readScalarVariable(mr, cxtVobSizeHeader);


            //nAuthor
           // int[] numAuthor = readVectorInt(mr, "nAuthor", false);
            //int[] NumAuthor = ReadArrayInt(mr, "nTitle");
            //int[] NumAuthor = ReadArrayInt(mr, "nContext", 0);

            //reading ss
            System.out.println("Reading ss....");
            outData.ss= readCellToArray(mr, "ss",true);

            //reading xx - context data
            System.out.println("\nReading xx....");
            outData.xx = readCellToSparseVector(mr,"xx",outData.contextDim,true);

            outData.ngroups=outData.ss.length;
            //Compute data size for each doc
            outData.numData=new int[ outData.ngroups];
            for (int jj=0;jj< outData.ngroups;jj++)
            {
                outData.numData[jj]=outData.ss[jj].length;
            }

        }catch (Exception e){
            e.printStackTrace();
        }


        System.out.println("\nDone loading Mat files!");
        return outData;
    }

    /**
     * Read variable with given name in matlab file
     * @param mr matlab file reader
     * @param variableHeaderName  variable name - scalar value
     * @return value of variable
     */
    public static double readScalarVariable(MatFileReader mr, String variableHeaderName)
    {
        MLDouble readToMLDouble = (MLDouble)mr.getMLArray(variableHeaderName);
        double[][] temp = readToMLDouble.getArray();
        double myVariable = temp[0][0];
        return myVariable;
    }

    /**
     * Read int vector variable in matlab file
     * @param mr matlab file reader
     * @param variableHeaderName variable header name
     * @param isMinusOne minus value in matlab format (from 1) to Java format (from 0)
     * @return vector of int values
     */

    public static int[] readVectorInt(MatFileReader mr, String variableHeaderName, boolean isMinusOne)
    {
        MLDouble tempnumdata = (MLDouble)mr.getMLArray(variableHeaderName);
        double[][] tnumdata = tempnumdata.getArray();
        int[] myVector = new int[tempnumdata.getSize()];
        int[] dim = tempnumdata.getDimensions();
        if (dim[0] > dim[1])
        {
            for (int jj = 0; jj < tempnumdata.getSize(); jj++)
            {
                myVector[jj] = (int)tnumdata[jj][0] - (isMinusOne?1:0);
            }
        }
        else
        {
            for (int jj = 0; jj < tempnumdata.getSize(); jj++)
            {
                myVector[jj] = (int)tnumdata[0][jj] - (isMinusOne?1:0);
            }
        }
        return myVector;
    }

    /**
     * Read a cell of value in matlab to SparseVector array
     * @param mr matlab file reader
     * @param MLArrayName variable header name
     * @param vocabSize vocabulary size of vector
     * @param isMinusOne minus value in matlab format (from 1) to Java format (from 0)
     * @return
     */
    public static SparseVector[] readCellToSparseVector(MatFileReader mr, String MLArrayName, int vocabSize, boolean isMinusOne)
    {
        MLCell cellxx = (MLCell)mr.getMLArray(MLArrayName);

        int ngroups = cellxx.getSize();
        SparseVector[] xx = new SparseVector[ngroups];



        for (int jj = 0; jj < ngroups; jj++)
        {
            MLCell tempVector = (MLCell)cellxx.get(jj);
            xx[jj] = new SparseVector(vocabSize);

            for (int ii = 0; ii < tempVector.getSize(); ii++)
            {
                MLDouble tempVal = (MLDouble)tempVector.get(ii);
                double[][] t = tempVal.getArray();
                int val = (int)t[0][0] - (isMinusOne?1:0);
                xx[jj].addValue(val,1);
            }
        }
        return xx;
    }

    /**
     * Read a cell of value in matlab to int array
     * @param mr matlab file reader
     * @param MLArrayName variable header name
     * @param isMinusOne minus value in matlab format (from 1) to Java format (from 0)
     * @return
     */
    public static int[][] readCellToArray(MatFileReader mr, String MLArrayName, boolean isMinusOne)//{cell}{cell} integer at w_ji
    {

        MLCell tempss = (MLCell)mr.getMLArray(MLArrayName);
        int ngroups = tempss.getSize();
        //output data
        int[][] ss = new int[ngroups][];

        for (int jj = 0; jj < ngroups; jj++)
        {
            MLCell tempVector = (MLCell)tempss.get(jj);
            ss[jj] = new int[tempVector.getSize()];
            for (int ii = 0; ii < tempVector.getSize(); ii++)
            {
                MLDouble tempVal = (MLDouble)tempVector.get(ii);
                double[][] t = tempVal.getArray();
                ss[jj][ii] = (int)t[0][0] - (isMinusOne?1:0);
            }
        }
        return ss;
    }
}
