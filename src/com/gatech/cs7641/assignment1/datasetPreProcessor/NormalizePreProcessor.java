package com.gatech.cs7641.assignment1.datasetPreProcessor;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class NormalizePreProcessor implements DatasetPreProcessor {

	@Override
	public Instances preProcessDataset(final Instances instances) {

		final Normalize normalizeFilter = new Normalize();

		try {
			return Filter.useFilter(instances, normalizeFilter);
		} catch (final Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

			throw new RuntimeException(e);
		}

	}

}
