package com.gatech.cs7641.assignment1.trainingRunner;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;
import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.google.common.collect.Sets;

public abstract class BaseTrainingRunner implements TrainingRunner {

	private final DatasetPartitioner partitioner;
	private final Instances originalTrainingSet;
	private final Instances testSet;
	private final AttributeSelector attrSelector;

	public BaseTrainingRunner(final AttributeSelector attrSelector,
			final DatasetPartitioner partitioner,
			final Instances originalTrainingSet, final Instances testSet) {
		super();
		this.partitioner = partitioner;
		this.originalTrainingSet = originalTrainingSet;
		this.testSet = testSet;
		this.attrSelector = attrSelector;
	}

	@Override
	public List<SingleRunResult> runTraining() {

		final List<SingleRunResult> toReturn = new ArrayList<SingleRunResult>();

		final ExecutorService execService = Executors.newFixedThreadPool(8);

		final List<Future<SingleRunResult>> listOfFutures = new ArrayList<Future<SingleRunResult>>();

		final Date start = new Date();
		System.out.println("Started at: "
				+ DateFormat.getDateTimeInstance(DateFormat.LONG,
						DateFormat.LONG).format(start));

		for (final AttributeSelectedInstances attributeSelectedInstances : attrSelector
				.getAttributeSelectedInstances(originalTrainingSet)) {

			final Iterable<Instances> trainingSets = partitioner
					.partitionDataset(attributeSelectedInstances
							.getAttributeSelectedInstances());

			final int[] selectedIndices = attributeSelectedInstances
					.getAttributeIndicesKeptFromOriginalInstance();

			final Instances trainingSetToEvaluateModelOn = attributeSelectedInstances
					.getAttributeSelectedInstances();

			final Instances testingSetToEvaluateModelOn = getPrunedInstances(
					testSet, selectedIndices);

			for (final Instances trainingInstances : trainingSets) {

				System.out.println("Now running on training set size of: "
						+ trainingInstances.numInstances());
				System.out.println("It has "
						+ trainingInstances.numAttributes()
						+ " attrs; original had "
						+ originalTrainingSet.numAttributes() + " attrs ");

				try {

					for (final ClassifierWithDescriptor cwd : buildClassifiers(trainingInstances)) {

						final Classifier classifier = cwd.getClassifier();

						final Future<SingleRunResult> future = execService
								.submit(new Callable<SingleRunResult>() {

									@Override
									public SingleRunResult call()
											throws Exception {

										System.out
												.println("Now running for classifier: "
														+ cwd.getDescriptor());

										final long start = System
												.currentTimeMillis();

										// get the error rates on training data
										final Evaluation trainingEval = new Evaluation(
												trainingInstances);
										trainingEval.evaluateModel(classifier,
												trainingSetToEvaluateModelOn);

										// error rates on testing data
										// make sure to only pick those
										// attributes from the testSet that the
										// trainingSet had.

										final Evaluation testingEval = new Evaluation(
												trainingInstances);
										testingEval.evaluateModel(classifier,
												testingSetToEvaluateModelOn);

										final long end = System
												.currentTimeMillis();

										// System.out.println("Classifier " +
										// cwd.getDescriptor() + " took " +
										// (end-start) + "ms");

										return new SingleRunResult(
												attributeSelectedInstances,
												trainingEval, testingEval, cwd);
									}

								});

						listOfFutures.add(future);
					}

				} catch (final Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
		}

		for (final Future<SingleRunResult> srr : listOfFutures) {
			try {
				toReturn.add(srr.get());
			} catch (final Exception e) {
				e.printStackTrace();
				throw new RuntimeException(e);
			}
		}

		final Date end = new Date();
		System.out.println("Ended at: "
				+ DateFormat.getDateTimeInstance(DateFormat.LONG,
						DateFormat.LONG).format(end));

		return toReturn;

	}

	private Instances getPrunedInstances(final Instances original,
			final int[] attributeIndicesToKeep) {

		final Instances toReturn = new Instances(original, 0,
				original.numInstances());

		final Set<Integer> allAttributeIndices = new HashSet<Integer>();
		for (int x = 0; x < original.numAttributes(); x++)
			allAttributeIndices.add(x);

		final Set<Integer> indicesToKeep = new HashSet<Integer>();
		for (int y = 0; y < attributeIndicesToKeep.length; y++)
			indicesToKeep.add(attributeIndicesToKeep[y]);

		final Set<Integer> attributeIndicesToRemove = Sets.difference(
				allAttributeIndices, indicesToKeep);

		final Integer[] asArray = attributeIndicesToRemove
				.toArray(new Integer[0]);
		Arrays.sort(asArray);

		int numPositionsToShiftDownwards = 0;
		for (final Integer i : asArray) {
			final int properIndexToDelete = i.intValue()
					- numPositionsToShiftDownwards;
			// System.out.println("toKeep was before: " + i + " but is now " +
			// properIndexToDelete);
			System.out.println("Now deleting index: " + properIndexToDelete
					+ "; original was: " + i.intValue()
					+ " and numDeletes before it was: "
					+ numPositionsToShiftDownwards);
			toReturn.deleteAttributeAt(properIndexToDelete);
			numPositionsToShiftDownwards++;
		}

		return toReturn;
	}

	protected abstract Iterable<ClassifierWithDescriptor> buildClassifiers(
			final Instances trainingInstances);

}
