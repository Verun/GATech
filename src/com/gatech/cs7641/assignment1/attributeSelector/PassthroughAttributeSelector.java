package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.Arrays;

import weka.core.Instances;

import com.google.common.collect.Iterables;

public class PassthroughAttributeSelector implements AttributeSelector {

	@Override
	public Iterable<InstancesWithSelectedIndices> getAttributeSelectedInstances(
			Instances original) {
		
		InstancesWithSelectedIndices toReturn = new InstancesWithSelectedIndices(original, null, original, null, null);
		
		return Iterables.unmodifiableIterable(Arrays.asList(toReturn));
	}

}
