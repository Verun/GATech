package com.gatech.cs7641.assignment1.attributeSelector;

import weka.core.Instances;

public class AttributeSelectedInstances {

	private final Instances attributeSelectedInstances;
	private final int[] attributeIndicesKeptFromOriginalInstance;
	private final Instances originalInstances;
	private final String evaluatorDescriptor;
	private final String searcherDescriptor;

	public AttributeSelectedInstances(
			final Instances attributeSelectedInstances,
			final int[] attributeIndicesKeptFromOriginalInstance,
			final Instances originalInstances,
			final String evaluatorDescriptor, final String searcherDescriptor) {
		super();
		this.attributeSelectedInstances = attributeSelectedInstances;
		this.attributeIndicesKeptFromOriginalInstance = attributeIndicesKeptFromOriginalInstance;
		this.originalInstances = originalInstances;
		this.evaluatorDescriptor = evaluatorDescriptor;
		this.searcherDescriptor = searcherDescriptor;
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

	public String getEvaluatorDescriptor() {
		return evaluatorDescriptor;
	}

	public String getSearcherDescriptor() {
		return searcherDescriptor;
	}
}
