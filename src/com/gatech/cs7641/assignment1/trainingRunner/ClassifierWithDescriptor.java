package com.gatech.cs7641.assignment1.trainingRunner;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class ClassifierWithDescriptor {

	private final Classifier classifier;
	private final String descriptor;
	private final Instances instancesTrainedOn;
	private final long timeItTookToTrain;

	public ClassifierWithDescriptor(final Classifier classifier,
			final String descriptor, final Instances instancesTrainedOn,
			final long timeItTookToTrain) {
		super();
		this.classifier = classifier;
		this.descriptor = descriptor;
		this.timeItTookToTrain = timeItTookToTrain;
		this.instancesTrainedOn = instancesTrainedOn;
	}

	public Classifier getClassifier() {
		return classifier;
	}

	public String getDescriptor() {
		return descriptor;
	}

	public Instances getInstancesTrainedOn() {
		return instancesTrainedOn;
	}

	public long getTimeItTookToTrain() {
		return timeItTookToTrain;
	}

}
