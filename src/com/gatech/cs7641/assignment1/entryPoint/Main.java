package com.gatech.cs7641.assignment1.entryPoint;

import java.util.Arrays;
import java.util.List;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.attributeSelector.ConfigurableAttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPartitioner.IncrementalPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.datasetPreProcessor.PassthroughPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.J48.J48TrainingRunner;
import com.google.common.collect.Iterables;

public class Main {

	private static final String ADULT_INCOME_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/adult.data.all.ed_num_removed.arff";
	private static final String CRIME_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/communities.data.arff";
	
	public static void main(String[] args) throws Exception {

		//load data:
		List<TrainingAndTestInstances> theData = getInstances();
		
		for (TrainingAndTestInstances trainingAndTestInstances : theData) {
		
			Instances training = trainingAndTestInstances.getTrainingInstances();
			Instances testing = trainingAndTestInstances.getTestInstances();
			
			System.out.println("Rel name: " + training.relationName() + ", Training set size: " + training.numInstances() + ", testing set size: " + testing.numInstances());
			
			//pass through preprocessor
			DatasetPreProcessor preProcessor = new PassthroughPreProcessor();
			
			//partitioner
			DatasetPartitioner partitioner = new IncrementalPartitioner(10);
			
			//TrainingRunner trainingRunner = new IBkTrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			TrainingRunner trainingRunner = new J48TrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			//TrainingRunner trainingRunner = new BoostedJ48TrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			
			List<SingleRunResult> evalHolder = trainingRunner.runTraining();
	
			ResultsDumper rd = new ResultsDumper();
			rd.dumpResultsToFile(evalHolder, "/Users/vrahimtoola/Desktop/" + "crime" + ".txt");
	
			//System.out.println(allResults);
		
		}
		

		
	}

	
	private static AttributeSelector getConfigurableAttributeSelector() {
		
		ASEvaluation[] evaluators = new ASEvaluation[] {(ASEvaluation)new CfsSubsetEval()};
		
		//attribute selector
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		BestFirst bestFirstSearch = new BestFirst();
		
		ASSearch[] searchers = new ASSearch[] {(ASSearch)search, (ASSearch)bestFirstSearch};
		
		AttributeSelector as = new ConfigurableAttributeSelector(Iterables.unmodifiableIterable(Arrays.asList(evaluators)), Iterables.unmodifiableIterable(Arrays.asList(searchers)), true);
		
		return as;
		
	}
	
		
	private static List<TrainingAndTestInstances> getInstances() throws Exception {
		//adult income data
		final String adultIncomeData = ADULT_INCOME_DATA;
		
		Instances beforeProcessing = DataSource.read(adultIncomeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);
		
		//System.out.println("Before processing, num instances: " + beforeProcessing.numInstances());

		//normalize numeric values, convert nominal attributes to binary
		Normalize normalize = new Normalize();
		
		NominalToBinary nominalToBinary = new NominalToBinary();
		nominalToBinary.setTransformAllValues(true);
		nominalToBinary.setBinaryAttributesNominal(false);
		
		MultiFilter mf = new MultiFilter();
		mf.setFilters(new Filter[] {normalize, nominalToBinary});
		mf.setInputFormat(beforeProcessing);
		
		Instances processed = Filter.useFilter(beforeProcessing, mf);
		
		Resample resample = new Resample(); //use the supervised version to maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);
		
		Instances adultTraining = Filter.useFilter(processed, resample);
		
		//DataSink.write("/Users/vrahimtoola/Desktop/adultTrain.arff", adultTraining);
		
		resample = new Resample(); //use the supervised version to maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);
		
		Instances adultTesting = Filter.useFilter(processed, resample);
		
		//////////////////////////////////
		final String crimeData = CRIME_DATA;
		
		beforeProcessing = DataSource.read(crimeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);
		
		//delete county, community name, fold attributes
		beforeProcessing.deleteAttributeAt(1);
		beforeProcessing.deleteAttributeAt(2);
		beforeProcessing.deleteAttributeAt(2);
	
		//discretize class values
		Discretize discretize = new Discretize();
		discretize.setAttributeIndices("last");
		discretize.setIgnoreClass(true);
		discretize.setBins(8);
		discretize.setUseEqualFrequency(true);
		
		//convert state attribute (index 0) to nominal
		NumericToNominal numericToNominal = new NumericToNominal();
		numericToNominal.setAttributeIndicesArray(new int[] {0});
		
		//normalize all numeric attributes
		normalize = new Normalize();
		
		mf = new MultiFilter();
		mf.setFilters(new Filter[] {discretize, numericToNominal, normalize});
		mf.setInputFormat(beforeProcessing);
		
		processed = Filter.useFilter(beforeProcessing, mf);
		
		//System.out.println("class index for crime is: " + processed.classIndex());
		
		//DataSink.write("/Users/vrahimtoola/Desktop/crimeAll.arff", processed);
		
		//
			
		resample = new Resample(); //use the supervised version to maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);
		
		Instances crimeTraining = Filter.useFilter(processed, resample);
		
		//DataSink.write("/Users/vrahimtoola/Desktop/adultTrain.arff", adultTraining);
		
		resample = new Resample(); //use the supervised version to maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);
		
		Instances crimeTesting = Filter.useFilter(processed, resample);
		
		return Arrays.asList(new TrainingAndTestInstances[] {
				new TrainingAndTestInstances(crimeTraining, crimeTesting),
				//new TrainingAndTestInstances(adultTraining, adultTesting),

				
		});
		
	}
	
	
	private static class TrainingAndTestInstances {
		
		private final Instances trainingInstances, testInstances;

		public TrainingAndTestInstances(Instances trainingInstances,
				Instances testInstances) {
			super();
			this.trainingInstances = trainingInstances;
			this.testInstances = testInstances;
		}

		public Instances getTrainingInstances() {
			return trainingInstances;
		}

		public Instances getTestInstances() {
			return testInstances;
		}
		
		
		
	}
}
