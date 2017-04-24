import org.bnpstat.util.DataField20NewsGroups;
import org.bnpstat.util.DataProcessing;
import spire.random.DistRng;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by hvhuynh on 5/2/2016.
 */
public class DataProcessingTest {
    public static void main(String[] args) {
        String vocFile = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\vocabulary.txt";
        HashMap vocList = DataProcessing.loadVocabulary(vocFile);
//
//
//
//        HashMap[] bowData = new HashMap[1];
//        String[] authors= new String[1];
//
//        String dataFile = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\20news-bydate\\20news-bydate-train\\alt.atheism\\49960";
//        DataField20NewsGroups data= DataProcessing.generateBOW20NewsGroups(dataFile, vocList);
//        bowData[0]=data.docBOW;
//        authors[0]=data.sender;
//        ArrayList docIds = new ArrayList();
//        docIds.add("1");
//        String outFile = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\out.txt";
//        String outFile1 = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\authors.txt";

//        try {
//            DataProcessing.writeBOWData(bowData, docIds, outFile);
//            DataProcessing.writeArrayData(authors,outFile1);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

//        String dataFolder = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\20news-bydate\\20news-bydate-train\\alt.atheism";
        String dataFolder = "N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\20news-bydate\\20news-bydate-test";

        File folder = new File(dataFolder);
        String outFolder="N:\\ResearchGroups\\PRaDA\\all-members\\datasets\\20NewsGroup\\20news-bydate";
        String outFile = outFolder+"\\bow.txt";
        String outFile1 = outFolder+"\\authors.txt";
        String outFile2 = outFolder+"\\doc_group_map.txt";

        System.out.println("Processing folder: ");
        for (File folderEntry : folder.listFiles()) {
            if (folderEntry.isDirectory()) {
                System.out.println("\t\t" +folderEntry);
                ArrayList docIds = new ArrayList();
                int nFile = folderEntry.listFiles().length;
                HashMap[] bowData = new HashMap[nFile];
                String[] authors = new String[nFile];
                String[] docGroups=new String[nFile];
                int count = 0;
                for (File fileEntry : folderEntry.listFiles()) {
                    if (!fileEntry.isDirectory()) {
                        DataField20NewsGroups data = DataProcessing.generateBOW20NewsGroups(folderEntry + "\\" + fileEntry.getName(), vocList);
                        bowData[count] = data.docBOW;
                        authors[count] = data.sender;
                        docGroups[count++]=folderEntry.getName();
                        docIds.add(fileEntry.getName());

                    }
                }
                try {
                    DataProcessing.writeBOWData(bowData, docIds, outFile, true);
                    DataProcessing.writeArrayData(authors, outFile1, true);
                    DataProcessing.writeArrayData(docGroups, outFile2, true);

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }


    }
}
