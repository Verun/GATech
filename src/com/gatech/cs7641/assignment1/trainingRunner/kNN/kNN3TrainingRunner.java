package com.gatech.cs7641.assignment1.trainingRunner.kNN;

import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;

public class kNN3TrainingRunner extends
		BaseTrainingRunner implements TrainingRunner {

	public kNN3TrainingRunner(int randSeed,
			List<DatasetPreProcessor> preProcessors,
			AttributeSelector attrSelector,
			DatasetPartitioner partitioner, Instances originalSet, Instances testSet) {
		super(randSeed, preProcessors, attrSelector, partitioner, originalSet, testSet);
		
	}

	@Override
	protected Classifier buildClassifier(Instances trainingInstances) {
		IBk kNN = new IBk();
		kNN.setKNN(3);
		try {
			kNN.buildClassifier(trainingInstances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return kNN;
	}

}
