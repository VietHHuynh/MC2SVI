package org.bnpstat.experiments;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.util.SystemClock;
import org.bnpstat.mc2.MC2InputDataMultCat;
import org.bnpstat.mc2.MC2StochasticVariationalInferenceSpark;
import org.bnpstat.stats.conjugacy.BayesianComponent;
import org.bnpstat.stats.conjugacy.GaussianGaussian;
import org.bnpstat.stats.conjugacy.MultinomialDirichlet;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Properties;
import java.util.Scanner;

/**
 * Created by hvhuynh on 11/20/2015.
 */
public class MC2ExperimentNIPS {
    public static void main(String args[]) {

        String userDir = System.getProperty("user.dir");
        System.setProperty("hadoop.home.dir", userDir + "\\hadoop-common-2.2.0-bin-master\\");

        System.out.println(System.getProperty("user.dir"));
        String currentDir = null;
        String propFileName = null;
        if (args.length == 0) {
            propFileName = "config.properties";
            System.out.println("Using the default setting file. Pass the configure file to use the user's config");
        } else if (args.length == 1) {
            propFileName = args[0];
        } else {
            System.out.println("Usage:java <classname>  or \n java <classname>  <configuration file> \n Please check \'config.properties\' file for template");
            System.exit(0);
        }

        Properties prop = new Properties();
        InputStream input = null;
        try {
            currentDir = new java.io.File(".").getCanonicalPath();

            //input = new FileInputStream(propFileName);

            // input=MC2ExperimentNIPSAuthorContext.class.getClassLoader().getResourceAsStream(propFileName);
            System.out.println(propFileName);
            input = new FileInputStream(propFileName);

            // load a properties file
            prop.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        double contentSym = Double.valueOf(prop.getProperty("mc2.contentDirichletSym"));
        double contextSym = Double.valueOf(prop.getProperty("mc2.contextDirichletSym"));
        int trunM = Integer.valueOf(prop.getProperty("mc2.trunM"));
        int trunK = Integer.valueOf(prop.getProperty("mc2.trunK"));
        int trunT = Integer.valueOf(prop.getProperty("mc2.trunT"));
        double aa = Double.valueOf(prop.getProperty("mc2.aa"));
        double ee = Double.valueOf(prop.getProperty("mc2.ee"));
        double vv = Double.valueOf(prop.getProperty("mc2.vv"));

        String contFilePath = prop.getProperty("mc2.contentPath");
        String cxtFilePath = prop.getProperty("mc2.contextPath");
        String metaFilepath = prop.getProperty("mc2.metaPath");
        String contextType = prop.getProperty("mc2.contextType");
        String outFolderPath = currentDir + File.separator + prop.getProperty("mc2.outFolderPath");
        System.out.println("Output folder" + outFolderPath);

        File outFolder = new File(outFolderPath);


        outFolderPath = outFolder.getPath();
        MC2InputDataMultCat data = new MC2InputDataMultCat();
        // Read meta data for content and context dimension
        try {
            input = new FileInputStream(metaFilepath);
            BufferedReader br = new BufferedReader(new InputStreamReader(input));
            data.contentDim = Integer.valueOf(br.readLine());
            data.contextDim = Integer.valueOf(br.readLine());

        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        System.out.println("Vocabulary Size = " + data.contentDim);
        System.out.println("Author Vocabulary Size = " + data.contextDim);

        // base measure of topic
        MultinomialDirichlet H = new MultinomialDirichlet(data.contentDim, contentSym * data.contentDim);
        BayesianComponent L = null;

        if (contextType.equals("Multinomial")) {
            // base measure of author
            L = new MultinomialDirichlet(data.contextDim, contextSym * data.contextDim);
        } else if (contextType.equals("Gaussian")) {
            L = new GaussianGaussian(0, 1, 1);
        } else {
            System.out.println("The context type is not defined yet which is Multinomial or Gaussian. Check mc2.contextType in config file!");
            System.exit(1);
        }


        System.out.println("Creating SVI inference engine...");
//        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ");
        SparkConf conf = new SparkConf().setAppName("MC2 Stochastic Variational Inference with Spark ").setMaster("local[*]");


        JavaSparkContext jsc = new JavaSparkContext(conf);

        MC2StochasticVariationalInferenceSpark mc2 = new MC2StochasticVariationalInferenceSpark(trunK, trunT, trunM, ee, aa, vv, L, H, jsc);
        try {
            mc2.loadData(contFilePath, cxtFilePath, true, contextType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Initializing ...");
        //mc2.initialize();
        mc2.initializeRandom();

        // SVI params
        int numIter = Integer.valueOf(prop.getProperty("mc2.numIter"));
        int batchSize = Integer.valueOf(prop.getProperty("mc2.batchSize"));
        double varrho = Double.valueOf(prop.getProperty("mc2.varrho"));
        double iota = Double.valueOf(prop.getProperty("mc2.iota"));

        System.out.println("Running ...");
        System.out.println("\tBatch size: " + batchSize);
        System.out.println("\tNo. of iterations: " + numIter);


        try {
            File outFile = new File(outFolderPath + File.separator + "config.properties");
            input = new FileInputStream(propFileName);

            if (outFile.exists()) {
                Scanner scanIn = new Scanner(System.in);
                System.out.print("The file \"" + outFile.toString() + "\" exists. \nDo you want to overwrite (Y/N)?");
                if (scanIn.nextLine().toUpperCase().equals("Y")) {
                    Files.copy(input, new File(outFolderPath + File.separator + "config.properties").toPath(), StandardCopyOption.REPLACE_EXISTING);
                } else
                    System.exit(0);
            } else {
                Files.copy(input, new File(outFolderPath + File.separator + "config.properties").toPath());

            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
        mc2.sviCategoryOptimizer(numIter, batchSize, varrho, iota, outFolderPath);
//        mc2.sviCategoryOptimizerNaiveMF(numIter, batchSize, varrho, iota, outFolderPath);


    }
}
