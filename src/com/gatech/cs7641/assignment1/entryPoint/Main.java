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
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.attributeSelector.ConfigurableAttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPartitioner.IncrementalPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.datasetPreProcessor.PassthroughPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.boostedJ48.BoostedJ48TrainingRunner;
import com.google.common.collect.Iterables;

public class Main {

	public static void main(String[] args) throws Exception {

		//load data:
		List<TrainingAndTestInstances> theData = getInstances();
		
		for (TrainingAndTestInstances trainingAndTestInstances : theData) {
		
			Instances training = trainingAndTestInstances.getTrainingInstances();
			Instances testing = trainingAndTestInstances.getTestInstances();
			
			System.out.println("Training set size: " + training.numInstances() + ", testing set size: " + testing.numInstances());
			
			//pass through preprocessor
			DatasetPreProcessor preProcessor = new PassthroughPreProcessor();
			
			//partitioner
			DatasetPartitioner partitioner = new IncrementalPartitioner(3);
			
			//TrainingRunner trainingRunner = new IBkTrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			//TrainingRunner trainingRunner = new J48TrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			TrainingRunner trainingRunner = new BoostedJ48TrainingRunner(1, Arrays.asList(preProcessor), getConfigurableAttributeSelector(), partitioner, training, testing);
			
			List<SingleRunResult> evalHolder = trainingRunner.runTraining();
	
			ResultsDumper rd = new ResultsDumper();
			String allResults = rd.getResults(evalHolder);
	
			System.out.println(allResults);
		
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
		final String adultIncomeData = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/adult.data.all.ed_num_removed.arff";
		
		Instances beforeProcessing = DataSource.read(adultIncomeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);
		
		System.out.println("Before processing, num instances: " + beforeProcessing.numInstances());

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
		
		resample = new Resample(); //use the supervised version to maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);
		
		Instances adultTesting = Filter.useFilter(processed, resample);
		
		return Arrays.asList(new TrainingAndTestInstances[] {
				new TrainingAndTestInstances(adultTraining, adultTesting)
				
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
