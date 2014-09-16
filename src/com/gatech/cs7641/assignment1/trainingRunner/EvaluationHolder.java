package com.gatech.cs7641.assignment1.trainingRunner;

import weka.classifiers.Evaluation;

public class EvaluationHolder {

	private final Iterable<Evaluation> trainingEvaluations;
	private final Iterable<Evaluation> testEvaluations;
	
	public EvaluationHolder(Iterable<Evaluation> trainingEvaluations,
			Iterable<Evaluation> testEvaluations) {
		super();
		this.trainingEvaluations = trainingEvaluations;
		this.testEvaluations = testEvaluations;
	}

	public Iterable<Evaluation> getTrainingEvaluations() {
		return trainingEvaluations;
	}

	public Iterable<Evaluation> getTestEvaluations() {
		return testEvaluations;
	}
	
	
}
