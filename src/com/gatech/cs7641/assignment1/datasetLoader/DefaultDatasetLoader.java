package com.gatech.cs7641.assignment1.datasetLoader;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DefaultDatasetLoader implements DatasetLoader {

	@Override
	public Instances loadDataset(String location) {
		try {
			return DataSource.read(location);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
			throw new RuntimeException(e);
		}
	}

}
