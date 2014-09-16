package com.gatech.cs7641.assignment1.trainingRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;

public abstract class BaseTrainingRunner implements TrainingRunner {

	private final List<DatasetPreProcessor> preProcessors;
	private final DatasetPartitioner partitioner;
	private final Instances originalSet;
	private final Instances testSet;
	private final int randSeed;
	
	public BaseTrainingRunner(int randSeed, List<DatasetPreProcessor> preProcessors, DatasetPartitioner partitioner, Instances originalSet, Instances testSet) {
		super();
		this.preProcessors = preProcessors;
		this.partitioner = partitioner;
		this.originalSet = originalSet;
		this.testSet = testSet;
		this.randSeed = randSeed;
	}

	@Override
	public EvaluationHolder runTraining() {

		Instances next = originalSet;
		
		next.randomize(new Random(randSeed));
		
		
			for (DatasetPreProcessor preProcessor : preProcessors) {
				
				next = preProcessor.preProcessDataset(next); 
				
			}
			
			Iterable<Instances> trainingSets = partitioner.partitionDataset(next);
			
			Classifier classifier = null;
			
			List<Evaluation> trainingEvaluations = new ArrayList<Evaluation>();
			List<Evaluation> testEvaluations = new ArrayList<Evaluation>();
			
			for (Instances trainingInstances : trainingSets) {
				
				System.out.println("Now running on training set size of: " + trainingInstances.numInstances());
				
				try {
					classifier = buildClassifier(trainingInstances);
					
					//get the error rates on training data
					Evaluation trainingEval = new Evaluation(trainingInstances);
					trainingEval.evaluateModel(classifier, trainingInstances);
					trainingEvaluations.add(trainingEval);
					
					//error rates on testing data
					Evaluation testingEval = new Evaluation(trainingInstances);
					testingEval.evaluateModel(classifier, testSet);
					testEvaluations.add(testingEval);
					
					
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			
			return new EvaluationHolder(trainingEvaluations, testEvaluations);
		
	}

	protected abstract Classifier buildClassifier(Instances trainingInstances);
	
}
