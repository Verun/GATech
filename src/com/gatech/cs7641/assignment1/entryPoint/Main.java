package com.gatech.cs7641.assignment1.entryPoint;

import java.util.Arrays;
import java.util.List;

import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.PassthroughAttributeSelector;
import com.gatech.cs7641.assignment1.datasetLoader.DatasetLoader;
import com.gatech.cs7641.assignment1.datasetLoader.DefaultDatasetLoader;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPartitioner.IncrementalPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.datasetPreProcessor.PassthroughPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.kNN.kNN3TrainingRunner;

public class Main {

	public static void main(String[] args) {

		//load data:
		DatasetLoader loader = new DefaultDatasetLoader();
		Instances initial = loader.loadDataset("/Users/vrahimtoola/Desktop/weka-3-6-11/data/glass.arff");
		initial.setClassIndex(9);
		//initial = initial.stringFreeStructure();
		
		//pass through preprocessor
		DatasetPreProcessor preProcessor = new PassthroughPreProcessor();
		
		//partitioner
		DatasetPartitioner partitioner = new IncrementalPartitioner(1, 5);
		
		TrainingRunner trainingRunner = new kNN3TrainingRunner(1, Arrays.asList(preProcessor), new PassthroughAttributeSelector(), partitioner, initial, initial);
		
		List<SingleRunResult> evalHolder = trainingRunner.runTraining();
		

		System.out.println("Now printing out eval on training sets...");
		for (SingleRunResult srr : evalHolder) {
				System.out.println(srr.getTrainingEvaluation().toSummaryString());
				System.out.println();
		}
		
		System.out.println("************************************");
		System.out.println("Now printing out eval on test set...");
		
		for (SingleRunResult srr : evalHolder) {
			System.out.println(srr.getTestEvaluation().toSummaryString());
			System.out.println();
		}
		
	}

}
