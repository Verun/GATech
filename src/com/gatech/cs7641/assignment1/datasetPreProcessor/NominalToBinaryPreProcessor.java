package com.gatech.cs7641.assignment1.datasetPreProcessor;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

public class NominalToBinaryPreProcessor implements DatasetPreProcessor {

	@Override
	public Instances preProcessDataset(Instances instances) {
		NominalToBinary nominalToBinary = new NominalToBinary();
		nominalToBinary.setTransformAllValues(true);
		nominalToBinary.setBinaryAttributesNominal(false);
		
		try {
			return Filter.useFilter(instances, nominalToBinary);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
			throw new RuntimeException(e);
		}
		
	}

}
