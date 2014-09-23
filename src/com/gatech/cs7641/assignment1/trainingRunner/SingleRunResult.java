package com.gatech.cs7641.assignment1.trainingRunner;

import weka.classifiers.Evaluation;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;

public class SingleRunResult {

	private final Evaluation trainingEvaluation;
	private final Evaluation testEvaluation;
	private final ClassifierWithDescriptor cwd;
	private final AttributeSelectedInstances asi;

	public SingleRunResult(final AttributeSelectedInstances asi,
			final Evaluation trainingEvaluation,
			final Evaluation testEvaluation,
			final ClassifierWithDescriptor classifierWithDescriptor) {
		super();
		this.asi = asi;
		this.trainingEvaluation = trainingEvaluation;
		this.testEvaluation = testEvaluation;
		cwd = classifierWithDescriptor;

	}

	public Evaluation getTrainingEvaluation() {
		return trainingEvaluation;
	}

	public Evaluation getTestEvaluation() {
		return testEvaluation;
	}

	public ClassifierWithDescriptor getClassifierWithDescriptor() {
		return cwd;
	}

	public AttributeSelectedInstances getAttributeSelectedInstances() {
		return asi;
	}

}
