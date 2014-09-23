package com.gatech.cs7641.assignment1.datasetPartitioner;

import weka.core.Instances;

import com.google.common.collect.AbstractIterator;

public class IncrementalPartitioner implements DatasetPartitioner {

	private final int numPartitions;

	public IncrementalPartitioner(final int numPartitions) {
		this.numPartitions = numPartitions;
	}

	@Override
	public Iterable<Instances> partitionDataset(final Instances instances) {

		final int numInstances = instances.numInstances();

		final int numInstancesPerPartition = numInstances / numPartitions;

		// step 2: return iterable of varying sized instances
		return new Iterable<Instances>() {

			@Override
			public AbstractIterator<Instances> iterator() {
				return new AbstractIterator<Instances>() {

					private int multiplier = 1;

					@Override
					protected Instances computeNext() {

						if (multiplier > numPartitions)
							return endOfData();

						final int numInstancesToCopyFromSource = (multiplier == numPartitions) ? numInstances
								: multiplier * numInstancesPerPartition;

						multiplier++;

						return new Instances(instances, 0,
								numInstancesToCopyFromSource);

					}

				};
			}

		};

	}

}
