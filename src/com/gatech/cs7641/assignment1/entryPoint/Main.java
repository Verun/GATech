package com.gatech.cs7641.assignment1.entryPoint;

import java.util.Arrays;
import java.util.List;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.attributeSelector.ConfigurableAttributeSelector;
import com.gatech.cs7641.assignment1.datasetLoader.DatasetLoader;
import com.gatech.cs7641.assignment1.datasetLoader.DefaultDatasetLoader;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPartitioner.IncrementalPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.datasetPreProcessor.PassthroughPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.kNN.kNN3TrainingRunner;
import com.google.common.collect.Iterables;

public class Main {

	public static void main(String[] args) {

		//load data:
		DatasetLoader loader = new DefaultDatasetLoader();
		Instances initial = loader.loadDataset("/Users/vrahimtoola/Desktop/weka-3-6-11/data/glass.arff");
		initial.setClassIndex(initial.numAttributes() - 1);
		//initial = initial.stringFreeStructure();
		
		//pass through preprocessor
		DatasetPreProcessor preProcessor = new PassthroughPreProcessor();
		
		//attribute selector
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		BestFirst bestFirstSearch = new BestFirst();
		AttributeSelector as = new ConfigurableAttributeSelector(Iterables.unmodifiableIterable(Arrays.asList((ASEvaluation)new CfsSubsetEval())), Iterables.unmodifiableIterable(Arrays.asList((ASSearch)search, (ASSearch)bestFirstSearch)));
		
		//partitioner
		DatasetPartitioner partitioner = new IncrementalPartitioner(1, 5);
		
		TrainingRunner trainingRunner = new kNN3TrainingRunner(1, Arrays.asList(preProcessor), as, partitioner, initial, initial);
		
		List<SingleRunResult> evalHolder = trainingRunner.runTraining();

		ResultsDumper rd = new ResultsDumper();
		String allResults = rd.getResults(evalHolder);
		
		System.out.println(allResults);
		
	}

}
