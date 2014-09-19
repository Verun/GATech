package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.Arrays;

import weka.core.Instances;

import com.google.common.collect.Iterables;

public class PassthroughAttributeSelector implements AttributeSelector {

	@Override
	public Iterable<AttributeSelectedInstances> getAttributeSelectedInstances(
			Instances original) {
		
		AttributeSelectedInstances toReturn = new AttributeSelectedInstances(original, getArrayOfAttributeIndices(original), original, "N/A", "N/A");
		
		return Iterables.unmodifiableIterable(Arrays.asList(toReturn));
	}
	
	private int[] getArrayOfAttributeIndices(Instances instances) {
		int numAttributes = instances.numAttributes();
		
		int[] toReturn = new int[numAttributes];
		for (int x = 0; x < numAttributes; x++) {
			toReturn[x] = x;
		}
		
		return toReturn;
	}

}
