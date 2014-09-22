package com.gatech.cs7641.assignment1.trainingRunner.boostedJ48;

import java.util.Iterator;
import java.util.List;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.entryPoint.GlobalConstants;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class BoostedJ48TrainingRunner extends BaseTrainingRunner {

	public BoostedJ48TrainingRunner(int randSeed,
			List<DatasetPreProcessor> preProcessors,
			AttributeSelector attrSelector, DatasetPartitioner partitioner,
			Instances trainingSet, Instances testSet) {
		super(attrSelector, partitioner, trainingSet, testSet);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected Iterable<ClassifierWithDescriptor> buildClassifiers(
			final Instances trainingInstances) {
		return new Iterable<ClassifierWithDescriptor>() {

			@Override
			public Iterator<ClassifierWithDescriptor> iterator() {

				return new AbstractIterator<ClassifierWithDescriptor>() {

					private int indexIntoMinNumObj = 0;
					private int indexIntoSubtreeRaising = 0;
					private int indexIntoNumIterationsArray = 0;
					
					private final int[] numIterationsArray = new int[] {1,2,3}; //too many iterations makes it too slow
					private final int[] minNumObjArray = new int[] {25, 50, 100};
					private final boolean[] subTreeRaisingArray = new boolean[] {true, false};
					
					private boolean returnedUnPruned = false;
					private boolean allClassifiersReturned = false;
					
					@Override
					protected ClassifierWithDescriptor computeNext() {
						
						if (allClassifiersReturned)
							return endOfData();
						
						AdaBoostM1 adaBoost = new AdaBoostM1();
						adaBoost.setSeed(GlobalConstants.RAND_SEED);
						J48 j48 = new J48();
						String descriptor = null;
						if (! returnedUnPruned) {
							
							j48.setUnpruned(true);
							returnedUnPruned = true;
							descriptor = "adaBoost (j48 unPruned)";

						} else {
						
							j48.setUnpruned(false);
							
							boolean subTreeRaisingFlag = getNextSubTreeRaisingFlag();
							int minNumObj = getNextMinNumObj();
							int numIterations = getNextNumIterations();
							
							j48.setSubtreeRaising(subTreeRaisingFlag);
							j48.setMinNumObj(minNumObj);
							adaBoost.setNumIterations(numIterations);
							
							indexIntoSubtreeRaising++;
							
							if (subTreeRaisingFlag == subTreeRaisingArray[subTreeRaisingArray.length - 1] &&
								minNumObj == minNumObjArray[minNumObjArray.length - 1] &&
								numIterations == numIterationsArray[numIterationsArray.length - 1])
								allClassifiersReturned = true;
							
							descriptor="adaBoost (j48 pruned);subTreeRaising=" + subTreeRaisingFlag + ";minNumObj=" + minNumObj + ";numIterations=" + numIterations;
						}
						
						adaBoost.setClassifier(j48);
						
							long trainingTime;
							try {
								long start = System.currentTimeMillis();
								adaBoost.buildClassifier(trainingInstances);
								trainingTime = System.currentTimeMillis() - start;
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
								
								throw new RuntimeException(e);
							}
							
							return new ClassifierWithDescriptor(adaBoost, descriptor, trainingInstances, trainingTime);

					}

					private int getNextNumIterations() {
						if (indexIntoNumIterationsArray >= numIterationsArray.length) {
							indexIntoNumIterationsArray = 0;
						}
						return numIterationsArray[indexIntoNumIterationsArray];
						
					}

					private int getNextMinNumObj() {
						if (indexIntoMinNumObj >= minNumObjArray.length) {
							indexIntoMinNumObj = 0;
							indexIntoNumIterationsArray++;
						}
						return minNumObjArray[indexIntoMinNumObj];
					}

					private boolean getNextSubTreeRaisingFlag() {
						if (indexIntoSubtreeRaising >= subTreeRaisingArray.length) {
							indexIntoSubtreeRaising = 0;
							indexIntoMinNumObj++;
						}
						return subTreeRaisingArray[indexIntoSubtreeRaising];
						
					}
					
				};
				
			}
			
		};	
	}

}
