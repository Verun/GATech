package com.gatech.cs7641.assignment1.entryPoint;

import java.util.Arrays;
import java.util.List;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.attributeSelector.ConfigurableAttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.datasetPartitioner.IncrementalPartitioner;
import com.gatech.cs7641.assignment1.datasetPreProcessor.DatasetPreProcessor;
import com.gatech.cs7641.assignment1.datasetPreProcessor.PassthroughPreProcessor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.J48.J48TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.boostedJ48.BoostedJ48TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.kNN.IBkTrainingRunner;
import com.google.common.collect.Iterables;

public class Main {

	private static final String ADULT_INCOME_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/adult.data.all.ed_num_removed.arff";
	private static final String CRIME_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/communities.data.arff";

	public static void main(final String[] args) throws Exception {

		// load data:
		final List<TrainingAndTestInstances> theData = getInstances();

		for (final TrainingAndTestInstances trainingAndTestInstances : theData) {

			final Instances training = trainingAndTestInstances
					.getTrainingInstances();
			final Instances testing = trainingAndTestInstances
					.getTestInstances();

			System.out.println("Rel name: " + training.relationName()
					+ ", Training set size: " + training.numInstances()
					+ ", testing set size: " + testing.numInstances());

			// pass through preprocessor
			final DatasetPreProcessor preProcessor = new PassthroughPreProcessor();

			// partitioner
			final DatasetPartitioner partitioner = new IncrementalPartitioner(
					10);

			final TrainingRunner[] trainingRunners = new TrainingRunner[] {
					new J48TrainingRunner(1, Arrays.asList(preProcessor),
							getConfigurableAttributeSelector(), partitioner,
							training, testing),
					new BoostedJ48TrainingRunner(1,
							Arrays.asList(preProcessor),
							getConfigurableAttributeSelector(), partitioner,
							training, testing),
					new IBkTrainingRunner(1, Arrays.asList(preProcessor),
							getConfigurableAttributeSelector(), partitioner,
							training, testing),

			};

			for (final TrainingRunner tr : trainingRunners) {

				final List<SingleRunResult> evalHolder = tr.runTraining();

				final ResultsDumper rd = new ResultsDumper();
				final String fileName = trainingAndTestInstances
						.getDatasetName() + "-" + tr.getDescriptor();
				rd.dumpResultsToFile(evalHolder, "/Users/vrahimtoola/Desktop/"
						+ fileName + ".txt");

			}

		}

	}

	private static AttributeSelector getConfigurableAttributeSelector() {

		final ASEvaluation[] evaluators = new ASEvaluation[] { new CfsSubsetEval() };

		// attribute selector
		final GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		final BestFirst bestFirstSearch = new BestFirst();

		final ASSearch[] searchers = new ASSearch[] { search, bestFirstSearch };

		final AttributeSelector as = new ConfigurableAttributeSelector(
				Iterables.unmodifiableIterable(Arrays.asList(evaluators)),
				Iterables.unmodifiableIterable(Arrays.asList(searchers)), true);

		return as;

	}

	private static List<TrainingAndTestInstances> getInstances()
			throws Exception {
		// adult income data
		final String adultIncomeData = ADULT_INCOME_DATA;

		Instances beforeProcessing = DataSource.read(adultIncomeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);

		// System.out.println("Before processing, num instances: " +
		// beforeProcessing.numInstances());

		// normalize numeric values, convert nominal attributes to binary
		Normalize normalize = new Normalize();

		final NominalToBinary nominalToBinary = new NominalToBinary();
		nominalToBinary.setTransformAllValues(true);
		nominalToBinary.setBinaryAttributesNominal(false);

		MultiFilter mf = new MultiFilter();
		mf.setFilters(new Filter[] { normalize, nominalToBinary });
		mf.setInputFormat(beforeProcessing);

		Instances processed = Filter.useFilter(beforeProcessing, mf);

		Resample resample = new Resample(); // use the supervised version to
											// maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);

		final Instances adultTraining = Filter.useFilter(processed, resample);

		// DataSink.write("/Users/vrahimtoola/Desktop/adultTrain.arff",
		// adultTraining);

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);

		final Instances adultTesting = Filter.useFilter(processed, resample);

		// ////////////////////////////////
		final String crimeData = CRIME_DATA;

		beforeProcessing = DataSource.read(crimeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);

		// delete county, community name, fold attributes
		beforeProcessing.deleteAttributeAt(1);
		beforeProcessing.deleteAttributeAt(2);
		beforeProcessing.deleteAttributeAt(2);

		// discretize class values
		final Discretize discretize = new Discretize();
		discretize.setAttributeIndices("last");
		discretize.setIgnoreClass(true);
		discretize.setBins(8);
		discretize.setUseEqualFrequency(true);

		// convert state attribute (index 0) to nominal
		final NumericToNominal numericToNominal = new NumericToNominal();
		numericToNominal.setAttributeIndicesArray(new int[] { 0 });

		// normalize all numeric attributes
		normalize = new Normalize();

		mf = new MultiFilter();
		mf.setFilters(new Filter[] { discretize, numericToNominal, normalize });
		mf.setInputFormat(beforeProcessing);

		processed = Filter.useFilter(beforeProcessing, mf);

		// System.out.println("class index for crime is: " +
		// processed.classIndex());

		// DataSink.write("/Users/vrahimtoola/Desktop/crimeAll.arff",
		// processed);

		//

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);

		final Instances crimeTraining = Filter.useFilter(processed, resample);

		// DataSink.write("/Users/vrahimtoola/Desktop/adultTrain.arff",
		// adultTraining);

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);

		final Instances crimeTesting = Filter.useFilter(processed, resample);

		return Arrays
				.asList(new TrainingAndTestInstances[] { new TrainingAndTestInstances(
						"crime", crimeTraining, crimeTesting),
				// new TrainingAndTestInstances("adult", adultTraining,
				// adultTesting),

				});

	}

	private static class TrainingAndTestInstances {

		private final Instances trainingInstances, testInstances;
		private final String datasetName;

		public TrainingAndTestInstances(final String datasetName,
				final Instances trainingInstances, final Instances testInstances) {
			super();
			this.trainingInstances = trainingInstances;
			this.testInstances = testInstances;
			this.datasetName = datasetName;
		}

		public Instances getTrainingInstances() {
			return trainingInstances;
		}

		public Instances getTestInstances() {
			return testInstances;
		}

		public String getDatasetName() {
			return datasetName;
		}

	}
}
