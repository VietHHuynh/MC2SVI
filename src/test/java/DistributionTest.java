import org.apache.commons.configuration.SystemConfiguration;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.bnpstat.stats.conjugacy.DoubleMultivariateData;
import org.bnpstat.stats.conjugacy.GaussianGaussian;
import org.bnpstat.stats.conjugacy.MultinomialDirichlet;
import org.bnpstat.stats.conjugacy.MultivariateGaussianGamma;
import org.nd4j.linalg.util.SynchronizedTable;

/**
 * Created by hvhuynh on 4/27/2016.
 */
public class DistributionTest {
    public static void main(String[] args) {
//        GaussianGaussian pp1 = new GaussianGaussian(2, 10, 2);
//        GaussianGaussian pp2 = (GaussianGaussian) pp1.clone();
//        NormalDistribution gauss = new NormalDistribution(4, 1);
//        int n1 = 100;
//        int n2 = 150;
//        double[] samples1 = new double[n1];
//        double[] samples2 = new double[n2];
//        samples1 = gauss.sample(n1);
//        samples2 = gauss.sample(n2);
//        GaussianGaussian pp3 = (GaussianGaussian) pp1.clone();
//
//        try {
//            for (int ii = 0; ii < n1; ii++) {
//                pp1.add(samples1[ii]);
//                pp3.add(samples1[ii]);
//            }
//            for (int ii = 0; ii < n2; ii++) {
//                pp2.add(samples2[ii]);
//                pp3.add(samples2[ii]);
//            }
//            System.out.println(" Distribution 1");
//            System.out.println(pp1);
//            System.out.println(" Distribution 2");
//            System.out.println(pp2);
//            pp1.plus(pp2);
//            System.out.println(" Distribution 1+2");
//            System.out.println(pp1);
//            System.out.println(" Distribution 3");
//            System.out.println(pp3);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        double[] mean={0.1, 1.2, 1.3,0.1};
        double[] mean1={0.2, 1, 1.2,0.4};
        double[] x={0, 1, 1,0};
        MultivariateGaussianGamma gau=new MultivariateGaussianGamma(mean,1,1,1);
        MultivariateGaussianGamma gau1=new MultivariateGaussianGamma(mean,1,1,1);
        try{
            System.out.println("Gaussian1 \n"+gau1);

            System.out.println("Gaussian before added a data point \n"+gau1);
            gau.add(new DoubleMultivariateData(x));
            System.out.println("Gaussian after added a data point\n"+gau);
            gau.stochasticUpdate(gau1,0.5);
            System.out.println("Gaussian after plus Gaussian1 \n"+gau);
        }catch(Exception e){
            e.printStackTrace();
        }

        MultinomialDirichlet mul=new MultinomialDirichlet(10,1);
        System.out.println("Created\n"+mul);
        try{
            mul.add(new Integer(2));
            mul.add(new Integer(4));
            mul.add(new Integer(9));
        }catch(Exception e){
            e.printStackTrace();
        }
        System.out.println("After added\n"+mul);

    }

}
