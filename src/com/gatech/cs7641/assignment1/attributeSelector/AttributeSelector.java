package com.gatech.cs7641.assignment1.attributeSelector;

import weka.core.Instances;

public interface AttributeSelector {

	Iterable<InstancesWithSelectedIndices> getAttributeSelectedInstances(Instances original);
}
