package com.gatech.cs7641.assignment1.trainingRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.attributeSelector.InstancesWithSelectedIndices;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;

public abstract class BaseTrainingRunner implements TrainingRunner {

	private final List<DatasetPreProcessor> preProcessors;
	private final DatasetPartitioner partitioner;
	private final Instances originalSet;
	private final Instances testSet;
	private final int randSeed;
	private final AttributeSelector attrSelector;
	
	public BaseTrainingRunner(int randSeed, List<DatasetPreProcessor> preProcessors, AttributeSelector attrSelector, DatasetPartitioner partitioner, Instances originalSet, Instances testSet) {
		super();
		this.preProcessors = preProcessors;
		this.partitioner = partitioner;
		this.originalSet = originalSet;
		this.testSet = testSet;
		this.randSeed = randSeed;
		this.attrSelector = attrSelector;
	}

	@Override
	public List<SingleRunResult> runTraining() {

		Instances next = originalSet;
		
		next.randomize(new Random(randSeed));	
		
			for (DatasetPreProcessor preProcessor : preProcessors) {
				
				next = preProcessor.preProcessDataset(next); 
				
			}
			
			/*
			 * for each instance (identified by eval, search, and selected indices) <-- collapse so that if the same indices are selected, only the first one runs)
	-partition into training sets
	-get classifiers (each identified by some string)
		for each training set, run training and test on test set, gather stats
		
	
instance (identified by eval, search, selected indices), classifier identifiers, training set size, training set stats..., test set stats
			 * 
			 * 
			 */
			
			List<SingleRunResult> toReturn = new ArrayList<SingleRunResult>();
			
			for (InstancesWithSelectedIndices attributeSelectedInstances : attrSelector.getAttributeSelectedInstances(next)) {
			
				Iterable<Instances> trainingSets = partitioner.partitionDataset(attributeSelectedInstances.getAttributeSelectedInstances());
				
				Classifier classifier = null;
			
				for (Instances trainingInstances : trainingSets) {
					
					System.out.println("Now running on training set size of: " + trainingInstances.numInstances());
					
					try {
						long start = System.currentTimeMillis();
						classifier = buildClassifier(trainingInstances);
						long timeToTrain = System.currentTimeMillis() - start;
						
						//get the error rates on training data
						Evaluation trainingEval = new Evaluation(trainingInstances);
						trainingEval.evaluateModel(classifier, trainingInstances);
										
						//error rates on testing data
						Evaluation testingEval = new Evaluation(trainingInstances);
						testingEval.evaluateModel(classifier, testSet);
						
						toReturn.add(new SingleRunResult(trainingEval, testingEval, timeToTrain));
						
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
				}
			}
			return toReturn;
		
	}

	protected abstract Classifier buildClassifier(Instances trainingInstances);
	
}
