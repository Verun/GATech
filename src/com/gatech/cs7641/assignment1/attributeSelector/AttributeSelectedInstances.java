package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.Arrays;

import weka.core.Instances;

public class AttributeSelectedInstances {

	private final Instances attributeSelectedInstances;
	private final int[] attributeIndicesKeptFromOriginalInstance;
	private final Instances originalInstances;
	private final String evaluator;
	private final String searcher;
	
	public AttributeSelectedInstances(Instances attributeSelectedInstances,
			int[] attributeIndicesKeptFromOriginalInstance,
			Instances originalInstances,
			String evaluator,
			String searcher) {
		super();
		this.attributeSelectedInstances = attributeSelectedInstances;
		this.attributeIndicesKeptFromOriginalInstance = attributeIndicesKeptFromOriginalInstance;
		this.originalInstances = originalInstances;
		this.evaluator = evaluator;
		this.searcher = searcher;
	}

	public Instances getAttributeSelectedInstances() {
		return attributeSelectedInstances;
	}

	public int[] getAttributeIndicesKeptFromOriginalInstance() {
		return attributeIndicesKeptFromOriginalInstance;
	}

	public Instances getInstancesBeforeAttributeSelection() {
		return originalInstances;
	}
	
	public String getEvaluatorDescriptorString() {
		return evaluator;
	}
	
	public String getSearcherDescriptorString() {
		return searcher;
	}
}