package com.gatech.cs7641.assignment1.datasetLoader;

import weka.core.Instances;

public interface DatasetLoader {
	Instances loadDataset(String location);
}
