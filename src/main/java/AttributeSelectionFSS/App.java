package AttributeSelectionFSS;


import jdk.swing.interop.SwingInterOpUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Collections;

public class App {

    static Instances data;
    static Instances test;
    static PrintWriter printData;
    static PrintWriter printTest;
    static String pathToSaveArff;
    static String pathToSaveModel;

    public static void main(String[] args) throws Exception{
        if(args.length != 6) {
            System.out.println("EZ DITUZU PARAMETROAK MODU EGOKIAN SARTU");
            System.out.println("1. PARAMETROA -> DATA ARFF FITXATEGIA");
            System.out.println("2. PARAMETROA -> LEHENENGO ARIKETAREN IRAGARPENAK GORDETZEKO FITXATEGIA");
            System.out.println("3. PARAMETROA -> TEST ARFF FITXATEGIA");
            System.out.println("4. PARAMETROA -> BIGARREN ARIKETAREN IRAGARPENAK GORDETZEKO FITXATEGIA");
            System.out.println("5. PARAMETROA -> HEADERS ZUZENTZEKO ERABILIKO DEN ARFF FITXATEGIA");
            System.out.println("6. PARAMETROA -> MODELO OPTIMOA GORDETZEKO FITXATEGIA");

        }else{
            ConverterUtils.DataSource sourceData = new ConverterUtils.DataSource(args[0]);
            data = sourceData.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            ConverterUtils.DataSource sourceTest = new ConverterUtils.DataSource(args[2]);
            test = sourceTest.getDataSet();
            test.setClassIndex(test.numAttributes()-1);

            FileWriter fileData = new FileWriter(args[1]);
            printData = new PrintWriter(fileData);

            FileWriter fileTest = new FileWriter(args[3]);
            printTest = new PrintWriter(fileTest);

            pathToSaveArff = args[4];

            pathToSaveModel = args[5];

            ataza1();
            ataza2();

            printData.close();
            printTest.close();
            fileData.close();
            fileTest.close();
        }
    }

    private static void ataza1() throws Exception {
        //Randomize
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        Instances random = Filter.useFilter(data,randomize);

        //Split
        RemovePercentage removeTrain = new RemovePercentage();
        removeTrain.setInputFormat(random);
        removeTrain.setInvertSelection(true);
        removeTrain.setPercentage(70);
        Instances holdTrain = Filter.useFilter(random,removeTrain);

        RemovePercentage removeTest = new RemovePercentage();
        removeTest.setInputFormat(random);
        removeTest.setInvertSelection(false);
        removeTest.setPercentage(70);
        Instances holdTest = Filter.useFilter(random,removeTest);

        //Train atributu hoberenak lortu
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setInputFormat(holdTrain);
        Instances trainBerria = Filter.useFilter(holdTrain,attributeSelection);

        //Instantzia berria sortu
        Instance instantziaBerria = new DenseInstance(trainBerria.numAttributes());
        trainBerria.add(instantziaBerria);

        //Estimazioak egin
        ataza1ReplaceGabe(trainBerria,holdTest);
        ataza1Replacearekin(trainBerria,holdTest);

    }

    private static void ataza1ReplaceGabe(Instances trainBerria, Instances holdTest) throws Exception{
        double percentage = (double)100/trainBerria.numInstances(); //instantzia bakar bat lortzeko portzentaia
        RemovePercentage trainOna = new RemovePercentage();
        trainOna.setInputFormat(trainBerria);
        trainOna.setInvertSelection(true);
        trainOna.setPercentage(100-percentage);
        Instances train = Filter.useFilter(trainBerria,trainOna);

        RemovePercentage bakarra = new RemovePercentage();
        bakarra.setInputFormat(trainBerria);
        bakarra.setInvertSelection(false);
        bakarra.setPercentage(100-percentage);
        Instances instantziaBakarra = Filter.useFilter(trainBerria,bakarra);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(instantziaBakarra);
        saver.setFile(new File(pathToSaveArff));
        saver.writeBatch();

        //modeloa entrenatu
        NaiveBayes naive = new NaiveBayes();
        naive.buildClassifier(train);
        SerializationHelper.write(pathToSaveModel, naive);

        holdTest = goiburuakZuzendu(holdTest);
        NaiveBayes naiveModel = (NaiveBayes) SerializationHelper.read(pathToSaveModel);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naiveModel, holdTest);
        printData.println("REPLACE EGIN GABE:");
        printData.println("IRAGARPENAK --> ");
        printData.println("WEIGHTED F-MEASURE -> " + eval.weightedFMeasure() + "\n");

    }

    private static void ataza1Replacearekin(Instances trainBerria, Instances holdTest) throws Exception {
        double percentage = (double)100/trainBerria.numInstances(); //instantzia bakar bat lortzeko portzentaia
        RemovePercentage trainOna = new RemovePercentage();
        trainOna.setInputFormat(trainBerria);
        trainOna.setInvertSelection(true);
        trainOna.setPercentage(100-percentage);
        Instances train = Filter.useFilter(trainBerria,trainOna);

        RemovePercentage bakarra = new RemovePercentage();
        bakarra.setInputFormat(trainBerria);
        bakarra.setInvertSelection(false);
        bakarra.setPercentage(100-percentage);
        Instances instantziaBakarra = Filter.useFilter(trainBerria,bakarra);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(instantziaBakarra);
        saver.setFile(new File(pathToSaveArff));
        saver.writeBatch();

        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(train);
        train = Filter.useFilter(train, replaceMissingValues);

        //modeloa entrenatu
        NaiveBayes naive = new NaiveBayes();
        naive.buildClassifier(train);
        SerializationHelper.write(pathToSaveModel, naive);

        holdTest = goiburuakZuzendu(holdTest);
        NaiveBayes naiveModel = (NaiveBayes) SerializationHelper.read(pathToSaveModel);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naiveModel, holdTest);
        printData.println("REPLACE EGIN ETA GERO:");
        printData.println("IRAGARPENAK --> ");
        printData.println("WEIGHTED F-MEASURE -> " + eval.weightedFMeasure() + "\n");

    }

    private static Instances goiburuakZuzendu(Instances test) throws Exception{
        Instances testBerria = test;
        ConverterUtils.DataSource path = new ConverterUtils.DataSource(pathToSaveArff);
        Instances egiaztatzeko = path.getDataSet();
        int i=0;
        for(Attribute attribute : Collections.list(test.enumerateAttributes())){
            if(!Collections.list(egiaztatzeko.enumerateAttributes()).contains(attribute)) {
                testBerria.deleteAttributeAt(attribute.index()-i);
                i++;
            }
        }
        return testBerria;
    }

    private static void ataza2() throws Exception {
        //Train atributu hoberenak lortu
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setInputFormat(data);
        Instances trainBerria = Filter.useFilter(data,attributeSelection);

        //Instantzia berria sortu
        Instance instantziaBerria = new DenseInstance(trainBerria.numAttributes());
        trainBerria.add(instantziaBerria);

        //Estimazioak egin
        ataza2ReplaceGabe(trainBerria);
        ataza2Replacearekin(trainBerria);
    }

    private static void ataza2ReplaceGabe(Instances trainBerria) throws Exception {
        double percentage = (double)100/trainBerria.numInstances(); //instantzia bakar bat lortzeko portzentaia
        RemovePercentage trainOna = new RemovePercentage();
        trainOna.setInputFormat(trainBerria);
        trainOna.setInvertSelection(true);
        trainOna.setPercentage(100-percentage);
        Instances train = Filter.useFilter(trainBerria,trainOna);

        RemovePercentage bakarra = new RemovePercentage();
        bakarra.setInputFormat(trainBerria);
        bakarra.setInvertSelection(false);
        bakarra.setPercentage(100-percentage);
        Instances instantziaBakarra = Filter.useFilter(trainBerria,bakarra);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(instantziaBakarra);
        saver.setFile(new File(pathToSaveArff));
        saver.writeBatch();

        NaiveBayes naive = new NaiveBayes();
        naive.buildClassifier(train);
        SerializationHelper.write(pathToSaveModel, naive);

        Instances testBerria = goiburuakZuzendu(test);
        NaiveBayes naiveModel = (NaiveBayes) SerializationHelper.read(pathToSaveModel);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naiveModel, testBerria);
        printTest.println("REPLACE EGIN ETA GERO:");
        printTest.println("IRAGARPENAK --> ");
        printTest.println("WEIGHTED F-MEASURE -> " + eval.weightedFMeasure() + "\n");

    }

    private static void ataza2Replacearekin(Instances trainBerria) throws Exception {
        double percentage = (double)100/trainBerria.numInstances(); //instantzia bakar bat lortzeko portzentaia
        RemovePercentage trainOna = new RemovePercentage();
        trainOna.setInputFormat(trainBerria);
        trainOna.setInvertSelection(true);
        trainOna.setPercentage(100-percentage);
        Instances train = Filter.useFilter(trainBerria,trainOna);

        RemovePercentage bakarra = new RemovePercentage();
        bakarra.setInputFormat(trainBerria);
        bakarra.setInvertSelection(false);
        bakarra.setPercentage(100-percentage);
        Instances instantziaBakarra = Filter.useFilter(trainBerria,bakarra);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(instantziaBakarra);
        saver.setFile(new File(pathToSaveArff));
        saver.writeBatch();

        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(train);
        train = Filter.useFilter(train, replaceMissingValues);

        NaiveBayes naive = new NaiveBayes();
        naive.buildClassifier(train);
        SerializationHelper.write(pathToSaveModel, naive);

        Instances testBerria = goiburuakZuzendu(test);
        NaiveBayes naiveModel = (NaiveBayes) SerializationHelper.read(pathToSaveModel);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naiveModel, testBerria);
        printTest.println("REPLACE EGIN ETA GERO:");
        printTest.println("IRAGARPENAK --> ");
        printTest.println("WEIGHTED F-MEASURE -> " + eval.weightedFMeasure() + "\n");

    }

}
