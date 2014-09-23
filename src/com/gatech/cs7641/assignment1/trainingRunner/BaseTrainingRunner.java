package com.gatech.cs7641.assignment1.trainingRunner;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;
import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.google.common.collect.Sets;

public abstract class BaseTrainingRunner implements TrainingRunner {

	private final DatasetPartitioner partitioner;
	private final Instances originalTrainingSet;
	private final Instances testSet;
	private final AttributeSelector attrSelector;
	
	public BaseTrainingRunner(AttributeSelector attrSelector, DatasetPartitioner partitioner, Instances originalTrainingSet, Instances testSet) {
		super();
		this.partitioner = partitioner;
		this.originalTrainingSet = originalTrainingSet;
		this.testSet = testSet;
		this.attrSelector = attrSelector;
	}

	@Override
	public List<SingleRunResult> runTraining() {
			
			List<SingleRunResult> toReturn = new ArrayList<SingleRunResult>();
	
			ExecutorService execService = Executors.newFixedThreadPool(8);
			
			List<Future<SingleRunResult>> listOfFutures = new ArrayList<Future<SingleRunResult>>();
			
			Date start = new Date();
			System.out.println("Started at: " + DateFormat.getDateTimeInstance(
		            DateFormat.LONG, DateFormat.LONG).format(start));
			
			for (final AttributeSelectedInstances attributeSelectedInstances : attrSelector.getAttributeSelectedInstances(originalTrainingSet)) {
			
				Iterable<Instances> trainingSets = partitioner.partitionDataset(attributeSelectedInstances.getAttributeSelectedInstances());
				
				final int[] selectedIndices = attributeSelectedInstances.getAttributeIndicesKeptFromOriginalInstance();
				
				final Instances trainingSetToEvaluateModelOn = attributeSelectedInstances.getAttributeSelectedInstances();
				
				final Instances testingSetToEvaluateModelOn = getPrunedInstances(testSet, selectedIndices);
				
				for (final Instances trainingInstances : trainingSets) {
					
					System.out.println("Now running on training set size of: " + trainingInstances.numInstances());
					System.out.println("It has " + trainingInstances.numAttributes() + " attrs; original had " + originalTrainingSet.numAttributes() + " attrs ");
					
					try {

						for (final ClassifierWithDescriptor cwd : buildClassifiers(trainingInstances)) {
						
							final Classifier classifier = cwd.getClassifier();
							
							Future<SingleRunResult> future = execService.submit(new Callable<SingleRunResult>() {

								@Override
								public SingleRunResult call() throws Exception {

									System.out.println("Now running for classifier: " + cwd.getDescriptor());
									
									long start = System.currentTimeMillis();
									
									//get the error rates on training data
									Evaluation trainingEval = new Evaluation(trainingInstances);
									trainingEval.evaluateModel(classifier, trainingSetToEvaluateModelOn);
													
									//error rates on testing data
									//make sure to only pick those attributes from the testSet that the trainingSet had.
									
									Evaluation testingEval = new Evaluation(trainingInstances);
									testingEval.evaluateModel(classifier, testingSetToEvaluateModelOn);
									
									long end = System.currentTimeMillis();
									
									//System.out.println("Classifier " + cwd.getDescriptor() + " took " + (end-start) + "ms");
									
									return new SingleRunResult(attributeSelectedInstances, trainingEval, testingEval, cwd);
								}
								
							});
							
							listOfFutures.add(future);
						}
						
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
				}
			}
			
			for (Future<SingleRunResult> srr : listOfFutures) {
				try {
					toReturn.add(srr.get());
				} catch (Exception e) {
					e.printStackTrace();
					throw new RuntimeException(e);
				}
			}
			
			
			Date end = new Date();
			System.out.println("Ended at: " + DateFormat.getDateTimeInstance(
		            DateFormat.LONG, DateFormat.LONG).format(end));
			
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
			System.out.println("Now deleting index: " + properIndexToDelete + "; original was: " + i.intValue() + " and numDeletes before it was: " + numPositionsToShiftDownwards);
			toReturn.deleteAttributeAt(properIndexToDelete);
			numPositionsToShiftDownwards++;
		}
		
		return toReturn;
	}
	
	protected abstract Iterable<ClassifierWithDescriptor> buildClassifiers(final Instances trainingInstances);
	
}
