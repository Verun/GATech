package com.gatech.cs7641.assignment1.trainingRunner.kNN;

import java.util.Iterator;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class kNN3TrainingRunner extends
		BaseTrainingRunner {

	public kNN3TrainingRunner(int randSeed,
			List<DatasetPreProcessor> preProcessors,
			AttributeSelector attrSelector,
			DatasetPartitioner partitioner, Instances originalSet, Instances testSet) {
		super(attrSelector, partitioner, originalSet, testSet);
		
	}

	@Override
	protected Iterable<ClassifierWithDescriptor> buildClassifiers(final Instances trainingInstances) {
		
		return new Iterable<ClassifierWithDescriptor>() {

			@Override
			public Iterator<ClassifierWithDescriptor> iterator() {

				return new AbstractIterator<ClassifierWithDescriptor>() {

					private boolean returned = false;
					
					@Override
					protected ClassifierWithDescriptor computeNext() {
						
						if (returned)
							return endOfData();
						
						IBk kNN = new IBk();
						kNN.setKNN(3);
						//kNN.
						long trainingTime;
						try {
							long start = System.currentTimeMillis();
							kNN.buildClassifier(trainingInstances);
							trainingTime = System.currentTimeMillis() - start;
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
							
							throw new RuntimeException(e);
						}
						
						returned = true;
						
						return new ClassifierWithDescriptor(kNN, "kNN=3", trainingInstances, trainingTime);
					}
					
				};
				
			}
			
		};	
		
	}

}
