package com.gatech.cs7641.assignment1.attributeSelector;

import weka.core.Instances;

public interface AttributeSelector {

	Iterable<AttributeSelectedInstances> getAttributeSelectedInstances(
			Instances original);
}
