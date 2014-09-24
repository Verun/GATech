package com.gatech.cs7641.assignment1.entryPoint;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
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
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;
import com.gatech.cs7641.assignment1.trainingRunner.TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.J48.J48TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.boostedJ48.BoostedJ48TrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.kNN.IBkTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.multiLayerPerceptron.MultiLayerPerceptronTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.svm.SVMTrainingRunner;
import com.google.common.collect.Iterables;

public class Main {

	private static String CARS_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/car.data.arff";
	private static String CRIME_DATA = "/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/communities.data.arff";

	public static void main(final String[] args) throws Exception {

		//System.out.println(System.getProperty("java.class.path"));
		
		String carsData = args[0];
		String crimeData = args[1];
		String outputDir = args[2];
		CARS_DATA = carsData;
		CRIME_DATA = crimeData;
		//String outputDir = "";
		
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

			// partitioner
			final DatasetPartitioner partitioner = new IncrementalPartitioner(
					10);

			final TrainingRunner[] trainingRunners = new TrainingRunner[] {
					new SVMTrainingRunner(getConfigurableAttributeSelector(), partitioner, training, testing),
					new J48TrainingRunner(
							getConfigurableAttributeSelector(), partitioner,
							training, testing),
					new BoostedJ48TrainingRunner(
							getConfigurableAttributeSelector(), partitioner,
							training, testing),
					new IBkTrainingRunner(
							getConfigurableAttributeSelector(), partitioner,
							training, testing),
					new MultiLayerPerceptronTrainingRunner(
							getConfigurableAttributeSelector(), partitioner,
							training, testing),


			};

			for (final TrainingRunner tr : trainingRunners) {

//				if (! tr.getDescriptor().equals("SVM") && trainingAndTestInstances.getDatasetName().equals("crime"))
//					continue;
				
				final List<SingleRunResult> evalHolder = tr.runTraining();

				final ResultsDumper rd = new ResultsDumper();
				final String fileName = trainingAndTestInstances
						.getDatasetName() + "-" + tr.getDescriptor();
				rd.dumpResultsToFile(evalHolder, outputDir + (outputDir.endsWith("/") ? "" : "/") 
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
		// cars data
		final String carsData = CARS_DATA;

		Instances beforeProcessing = DataSource.read(carsData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);

		// System.out.println("Before processing, num instances: " +
		// beforeProcessing.numInstances());

		final NominalToBinary nominalToBinary = new NominalToBinary();
		nominalToBinary.setTransformAllValues(true);
		nominalToBinary.setBinaryAttributesNominal(false);

		MultiFilter mf = new MultiFilter();
		mf.setFilters(new Filter[] { nominalToBinary });
		mf.setInputFormat(beforeProcessing);

		Instances processed = Filter.useFilter(beforeProcessing, mf);

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/carsAll.arff",
//		 processed);
		
		Resample resample = new Resample(); // use the supervised version to
											// maintain the class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);

		final Instances carsTraining = Filter.useFilter(processed, resample);
		carsTraining.randomize(new Random(GlobalConstants.RAND_SEED));

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/carsTrain.arff",
//		 carsTraining);

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);

		final Instances carsTesting = Filter.useFilter(processed, resample);

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/carsTest.arff",
//		 carsTesting);
		
		// ////////////////////////////////
		final String crimeData = CRIME_DATA;

		beforeProcessing = DataSource.read(crimeData);
		beforeProcessing.setClassIndex(beforeProcessing.numAttributes() - 1);

		// delete county, community, community name, fold attributes
		beforeProcessing.deleteAttributeAt(1);
		beforeProcessing.deleteAttributeAt(1);
		beforeProcessing.deleteAttributeAt(1);
		beforeProcessing.deleteAttributeAt(1);

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
		Normalize normalize = new Normalize();

		mf = new MultiFilter();
		mf.setFilters(new Filter[] { discretize, numericToNominal, normalize });
		mf.setInputFormat(beforeProcessing);

		processed = Filter.useFilter(beforeProcessing, mf);

		// System.out.println("class index for crime is: " +
		// processed.classIndex());

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/crimeAll.arff",
//		 processed);

		//

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInputFormat(processed);

		final Instances crimeTraining = Filter.useFilter(processed, resample);
		crimeTraining.randomize(new Random(GlobalConstants.RAND_SEED));

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/crimeTrain.arff",
//		 crimeTraining);

		resample = new Resample(); // use the supervised version to maintain the
									// class distribution
		resample.setNoReplacement(true);
		resample.setRandomSeed(GlobalConstants.RAND_SEED);
		resample.setSampleSizePercent(100 - GlobalConstants.HOLDOUT_PERCENTAGE);
		resample.setInvertSelection(true);
		resample.setInputFormat(processed);

		final Instances crimeTesting = Filter.useFilter(processed, resample);

//		 DataSink.write("/Users/vrahimtoola/Desktop/GATech/Assignment 1/Data/JavaInputs/crimeTest.arff",
//		 crimeTesting);
		
		return Arrays
				.asList(new TrainingAndTestInstances[] { 

				 new TrainingAndTestInstances("cars", carsTraining,
				 carsTesting),
					new TrainingAndTestInstances(
					"crime", crimeTraining, crimeTesting),

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
