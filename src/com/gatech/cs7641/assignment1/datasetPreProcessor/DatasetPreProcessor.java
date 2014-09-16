package com.gatech.cs7641.assignment1.datasetPreProcessor;

import weka.core.Instances;

public interface DatasetPreProcessor {
	Instances preProcessDataset(Instances instances);
}
