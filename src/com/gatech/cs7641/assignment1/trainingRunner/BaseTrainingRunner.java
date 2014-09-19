package com.gatech.cs7641.assignment1.trainingRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;
import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

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
			
			for (AttributeSelectedInstances attributeSelectedInstances : attrSelector.getAttributeSelectedInstances(next)) {
			
				Iterable<Instances> trainingSets = partitioner.partitionDataset(attributeSelectedInstances.getAttributeSelectedInstances());
				
				Classifier classifier = null;
			
				for (Instances trainingInstances : trainingSets) {
					
					System.out.println("Now running on training set size of: " + trainingInstances.numInstances());
					System.out.println("It has " + trainingInstances.numAttributes() + " num attributes; original had " + next.numAttributes() + " attrs ");
					
					try {
						for (ClassifierWithDescriptor cwd : buildClassifiers(trainingInstances)) {
						
							classifier = cwd.getClassifier();
							
							//get the error rates on training data
							Evaluation trainingEval = new Evaluation(trainingInstances);
							trainingEval.evaluateModel(classifier, trainingInstances);
											
							//error rates on testing data
							//make sure to only pick those attributes from the testSet that the trainingSet had.
							
							Evaluation testingEval = new Evaluation(trainingInstances);
							testingEval.evaluateModel(classifier, getPrunedInstances(testSet, attributeSelectedInstances.getAttributeIndicesKeptFromOriginalInstance()));
							
							//attributeSelectedInstances = same size as original training set, but with possibly
							//a subset of the attributes.
							//trainingEval - training evaluation results on a partition of the attribute selected instances.
							//testingEval - testing evaluation results on the original test set.
							//classifierWithDescriptor - info on the classifier used. use this to get to the training set and training time for the
							//classifier.
							toReturn.add(new SingleRunResult(attributeSelectedInstances, trainingEval, testingEval, cwd));
						}
						
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
				}
			}
			return toReturn;
		
	}

	private Instances getPrunedInstances(Instances original, int[] attributeIndicesToKeep) {
		
		Instances toReturn = new Instances(original, 0, original.numInstances());
		
		Set<Integer> allAttributeIndices = new HashSet<Integer>();
		for (int x = 0; x < original.numAttributes(); x++)
			allAttributeIndices.add(x);
		
		Set<Integer> indicesToKeep = new HashSet<Integer>();
		for (int y = 0; y < attributeIndicesToKeep.length; y++)
			indicesToKeep.add(attributeIndicesToKeep[y]);
		
		Set<Integer> attributeIndicesToRemove = Sets.difference(allAttributeIndices, indicesToKeep);
		
		Integer[] asArray = attributeIndicesToRemove.toArray(new Integer[0]);
		Arrays.sort(asArray);
		
		int numPositionsToShiftDownwards = 0;
		for (Integer i : asArray) {
			int properIndexToDelete = i.intValue() - numPositionsToShiftDownwards;
			//System.out.println("toKeep was before: " + i + " but is now " + properIndexToDelete);
			toReturn.deleteAttributeAt(properIndexToDelete);
			numPositionsToShiftDownwards++;
		}
		
		return toReturn;
	}
	
	protected abstract Iterable<ClassifierWithDescriptor> buildClassifiers(final Instances trainingInstances);
	
}
