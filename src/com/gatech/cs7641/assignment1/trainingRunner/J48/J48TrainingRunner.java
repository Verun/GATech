package com.gatech.cs7641.assignment1.trainingRunner.J48;

import java.util.Iterator;
import java.util.List;

import weka.classifiers.trees.J48;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class J48TrainingRunner extends BaseTrainingRunner {

	public J48TrainingRunner(int randSeed,
			List<DatasetPreProcessor> preProcessors,
			AttributeSelector attrSelector, DatasetPartitioner partitioner,
			Instances trainingSet, Instances testSet) {
		super(randSeed, preProcessors, attrSelector, partitioner, trainingSet, testSet);
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
					
					private final int[] minNumObjArray = new int[] {3, 10, 25};
					private final boolean[] subTreeRaisingArray = new boolean[] {true, false};
					
					private boolean returnedUnPruned = false;
					
					@Override
					protected ClassifierWithDescriptor computeNext() {
						
						J48 j48 = new J48();
						String descriptor = null;
						if (! returnedUnPruned) {
							j48.setUnpruned(true);
							returnedUnPruned = true;
							descriptor = "unPruned";

						} else {
						
							j48.setUnpruned(false);
							
							if (indexIntoSubtreeRaising >= subTreeRaisingArray.length) {
								indexIntoSubtreeRaising = 0;
								indexIntoMinNumObj++;
							}
							
							if (indexIntoMinNumObj >= minNumObjArray.length) {
								return endOfData();
							}
								
							boolean subTreeRaising = subTreeRaisingArray[indexIntoSubtreeRaising];
							indexIntoSubtreeRaising++;
							
							int minNumObj = minNumObjArray[indexIntoMinNumObj];
							
							j48.setSubtreeRaising(subTreeRaising);
							j48.setMinNumObj(minNumObj);
							
							descriptor="subTreeRaising=" + subTreeRaising + ";minNumObj=" + minNumObj;
						}
						
							long trainingTime;
							try {
								long start = System.currentTimeMillis();
								j48.buildClassifier(trainingInstances);
								trainingTime = System.currentTimeMillis() - start;
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
								
								throw new RuntimeException(e);
							}
							
							return new ClassifierWithDescriptor(j48, descriptor, trainingInstances, trainingTime);

					}
					
				};
				
			}
			
		};	
		
	}

}