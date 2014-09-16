package com.gatech.cs7641.assignment1.datasetPartitioner;

import weka.core.Instances;

public interface DatasetPartitioner {
	Iterable<Instances> partitionDataset(Instances instances);
}
