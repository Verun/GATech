package com.gatech.cs7641.assignment1.trainingRunner.kNN;

import java.util.Iterator;
import java.util.List;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class IBkTrainingRunner extends BaseTrainingRunner {

	public IBkTrainingRunner(
			final AttributeSelector attrSelector,
			final DatasetPartitioner partitioner, final Instances trainingSet,
			final Instances testSet) {
		super(attrSelector, partitioner, trainingSet, testSet);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected Iterable<ClassifierWithDescriptor> buildClassifiers(
			final Instances trainingInstances) {

		final int[] valuesOfK = new int[] { 1, 5, 20, 1000 };

		return new Iterable<ClassifierWithDescriptor>() {

			@Override
			public Iterator<ClassifierWithDescriptor> iterator() {

				return new AbstractIterator<ClassifierWithDescriptor>() {

					private int indexIntoValuesOfK = 0;
					private int indexIntoWeightTags = 0;

					@Override
					protected ClassifierWithDescriptor computeNext() {

						while (true) {

							if (indexIntoWeightTags >= IBk.TAGS_WEIGHTING.length) {
								indexIntoWeightTags = 0;
								indexIntoValuesOfK++;
							}

							if (indexIntoValuesOfK >= valuesOfK.length) {
								return endOfData();
							}

							final Tag distanceWeightTag = IBk.TAGS_WEIGHTING[indexIntoWeightTags];
							indexIntoWeightTags++;

							final int k = valuesOfK[indexIntoValuesOfK];

							if (k == 1
									&& distanceWeightTag.getID() != IBk.WEIGHT_NONE)
								continue; // can only use kNN > 1 with
											// similarity/inverse weight
											// distancing

							final SelectedTag selectedTag = new SelectedTag(
									distanceWeightTag.getID(),
									IBk.TAGS_WEIGHTING);

							final IBk kNN = new IBk();
							kNN.setKNN(k);
							kNN.setDistanceWeighting(selectedTag);
							// kNN.setNearestNeighbourSearchAlgorithm(nearestNeighbourSearchAlgorithm);

							long trainingTime;
							try {
								final long start = System.currentTimeMillis();
								kNN.buildClassifier(trainingInstances);
								trainingTime = System.currentTimeMillis()
										- start;
							} catch (final Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();

								throw new RuntimeException(e);
							}

							return new ClassifierWithDescriptor(kNN, "kNN=" + k
									+ ";distance weighting="
									+ selectedTag.getSelectedTag().getIDStr(),
									trainingInstances, trainingTime);

						}
					}

				};

			}

		};

	}

	@Override
	public String getDescriptor() {
		return "KNN";
	}

}
