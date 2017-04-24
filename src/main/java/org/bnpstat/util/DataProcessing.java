package org.bnpstat.util;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;
import java.util.Map.*;

/**
 * Created by hvhuynh on 5/2/2016.
 */
public class DataProcessing {
    public static HashMap<String, Integer> loadVocabulary(String filePath) {
        HashMap<String, Integer> vocList = new HashMap<String, Integer>();
        String line;
        int count=0; // starting index for vocab
        try {
            InputStream fis = new FileInputStream(filePath);
            InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
            BufferedReader br = new BufferedReader(isr);
            while ((line = br.readLine()) != null) {
                vocList.put(line, count++);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return vocList;
    }

    public static String mapToSVMLibString(HashMap<Integer, Integer> map) {
        StringBuilder stringBuilder = new StringBuilder();
        String sep = " ";

        SortedSet<Integer> keys = new TreeSet<Integer>(map.keySet());
        for (Integer key : keys) {
            stringBuilder.append(key+ ":" + map.get(key) + sep);
        }
        if(stringBuilder.length()>2)
        stringBuilder.delete(stringBuilder.length() - 1, stringBuilder.length());
        return stringBuilder.toString();
    }

    public static void writeBOWData(HashMap<Integer, Integer>[] bow, List<String> docIds, String fileName,boolean append) throws Exception {

        int nDocs = docIds.size();
        if (nDocs != bow.length)
            throw new Exception("The sizes of bag of words mapping and document IDs are not equal!");
        try {
            OutputStream fos = new FileOutputStream(fileName,append);
            OutputStreamWriter osw = new OutputStreamWriter(fos, Charset.forName("UTF-8"));
            BufferedWriter bw = new BufferedWriter(osw);
            for (int ii = 0; ii < bow.length; ii++) {
                bw.write(docIds.get(ii)+" "+mapToSVMLibString(bow[ii]));
                bw.newLine();
            }
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static HashMap<Integer, Integer> generateBOW(String filePath,HashMap<String,Integer>vocabList){
        HashMap<Integer, Integer> bowMap = new HashMap();
        String line;
        try {
            InputStream fis = new FileInputStream(filePath);
            InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
            BufferedReader br = new BufferedReader(isr);
            while ((line = br.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(line);
                while (st.hasMoreTokens()) {
                    String word=st.nextToken();
                    if(vocabList.containsKey(word)) {
                        Integer val = bowMap.get(word);
                        int count = val==null ? 0 : val;
                        count+=1;
                        bowMap.put(vocabList.get(word),count);
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return bowMap;
    }
    public static DataField20NewsGroups generateBOW20NewsGroups(String filePath,HashMap<String,Integer>vocabList){
        DataField20NewsGroups result=new DataField20NewsGroups();
        result.docBOW = new HashMap();
        String line;
        try {
            InputStream fis = new FileInputStream(filePath);
            InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
            BufferedReader br = new BufferedReader(isr);
            boolean flag=false;
            while ((line = br.readLine()) != null) {
                if(line.contains("From:")&&!flag) {
                    result.sender=line;
                    flag=true;
                }

                StringTokenizer st = new StringTokenizer(line);
                while (st.hasMoreTokens()) {
                    String word=st.nextToken();
                    if(vocabList.containsKey(word)) {
                        Integer val = result.docBOW.get(vocabList.get(word));
                        int count = val==null ? 0 : val;
                        count+=1;
                        result.docBOW.put(vocabList.get(word),count);
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
    public static void writeArrayData(String[] list,String fileName,boolean append){
        try {
            OutputStream fos = new FileOutputStream(fileName,append);
            OutputStreamWriter osw = new OutputStreamWriter(fos, Charset.forName("UTF-8"));
            BufferedWriter bw = new BufferedWriter(osw);
            for (int ii = 0; ii < list.length; ii++) {
                bw.write(list[ii]);
                bw.newLine();
            }
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



}