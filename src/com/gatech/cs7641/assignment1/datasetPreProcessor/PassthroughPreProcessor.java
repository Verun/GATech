package com.gatech.cs7641.assignment1.datasetPreProcessor;

import weka.core.Instances;

public class PassthroughPreProcessor implements DatasetPreProcessor {

	@Override
	public Instances preProcessDataset(final Instances instances) {
		return instances;
	}

}
