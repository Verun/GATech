package com.gatech.cs7641.assignment1.datasetPreProcessor;

import java.util.List;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;

public class MetaFilterPreProcessor implements DatasetPreProcessor {

	private final List<Filter> filtersToApply;

	public MetaFilterPreProcessor(final List<Filter> filtersToApply) {
		super();
		this.filtersToApply = filtersToApply;
	}

	@Override
	public Instances preProcessDataset(final Instances instances) {
		final MultiFilter mf = new MultiFilter();
		mf.setFilters(filtersToApply.toArray(new Filter[0]));

		try {
			return Filter.useFilter(instances, mf);
		} catch (final Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

			throw new RuntimeException(e);
		}
	}

}
