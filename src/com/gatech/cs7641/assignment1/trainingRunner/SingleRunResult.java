package com.gatech.cs7641.assignment1.trainingRunner;

import weka.classifiers.Evaluation;

public class SingleRunResult {

	private final Evaluation trainingEvaluation;
	private final Evaluation testEvaluation;
	private final long numMillisecondsToBuildClassifier;
	
	public SingleRunResult(Evaluation trainingEvaluation,
			Evaluation testEvaluation, long numMillisecondsToBuildClassifier) {
		super();
		this.trainingEvaluation = trainingEvaluation;
		this.testEvaluation = testEvaluation;
		this.numMillisecondsToBuildClassifier = numMillisecondsToBuildClassifier;
	}

	public Evaluation getTrainingEvaluation() {
		return trainingEvaluation;
	}

	public Evaluation getTestEvaluation() {
		return testEvaluation;
	}

	public long getNumMillisecondsToBuildClassifier() {
		return numMillisecondsToBuildClassifier;
	}
	
	
	
	
}
