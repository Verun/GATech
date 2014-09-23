package com.gatech.cs7641.assignment1.trainingRunner.multiLayerPerceptron;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.entryPoint.GlobalConstants;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class MultiLayerPerceptronTrainingRunner extends BaseTrainingRunner {

	public MultiLayerPerceptronTrainingRunner(AttributeSelector attrSelector,
			DatasetPartitioner partitioner, Instances originalTrainingSet,
			Instances testSet) {
		super(attrSelector, partitioner, originalTrainingSet, testSet);
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getDescriptor() {
		return "MultiLayerPerceptron";
	}

	@Override
	protected Iterable<ClassifierWithDescriptor> buildClassifiers(
			final Instances trainingInstances) {

		return new Iterable<ClassifierWithDescriptor>() {

			@Override
			public Iterator<ClassifierWithDescriptor> iterator() {

				return new AbstractIterator<ClassifierWithDescriptor>() {

					private int indexIntoModelParamsList = 0;
					private final List<ModelParams> modelParamsList = new ArrayList<ModelParams>();
					
					{
						//instance initializer
						float[] learningRates = new float[] {0.1f, 0.3f, 0.5f};			
						float[] momentums = new float[] {0.1f, 0.2f, 0.3f};
						String[] hiddenLayers = new String[] {"a,b"};
						int[] trainingEpochs = new int[] {100, 200};
						int[] validationSetSizes = new int[] {10, 20};
						
						for (float l : learningRates)
							for (float m : momentums)
								for (String h : hiddenLayers)
									for (int t : trainingEpochs)
										for (int v : validationSetSizes)
											modelParamsList.add(new ModelParams(l, m, t, v, h));
						
					}
					
					@Override
					protected ClassifierWithDescriptor computeNext() {

						Classifier toTrain = null;
						String descriptor = "";
						
						if (indexIntoModelParamsList >= modelParamsList.size())
							return endOfData();
						
						MultilayerPerceptron mlp = new MultilayerPerceptron();
						
						//training sets have already been appropriately pre-processed, so turn these off
						mlp.setNominalToBinaryFilter(false);
						mlp.setNormalizeAttributes(false);
						
						mlp.setReset(true); //let it modify it's own learning rate
						
						ModelParams mp = modelParamsList.get(indexIntoModelParamsList++);
					
						mlp.setLearningRate(mp.getLearningRate());
						mlp.setMomentum(mp.getMomentum());
						mlp.setHiddenLayers(mp.getComputedHiddenLayerString(trainingInstances));
						mlp.setSeed(GlobalConstants.RAND_SEED);
						mlp.setTrainingTime(mp.getTrainingEpochs());
						mlp.setValidationSetSize(mp.getValidationSetSize());
						
						descriptor=	"l="+mp.getLearningRate() + 
									";m="+mp.getMomentum() + 
									";h="+mp.getComputedHiddenLayerString(trainingInstances) +
									";t="+mp.getTrainingEpochs() +
									";v="+mp.getValidationSetSize();
						
						long trainingTime;
						try {
							toTrain = mlp;
							final long start = System.currentTimeMillis();
							toTrain.buildClassifier(trainingInstances);
							trainingTime = System.currentTimeMillis() - start;
						} catch (final Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();

							throw new RuntimeException(e);
						}

						return new ClassifierWithDescriptor(toTrain, descriptor,
								trainingInstances, trainingTime);

					}					
					
				};

			}

		};
		
	}
	
	private static class ModelParams {
		
		private final float learningRate, momentum;
		private final int trainingEpochs, validationSetSize;
		private final String hiddenLayerString;
		
		public ModelParams(float learningRate, float momentum,
				int trainingEpochs, int validationSetSize,
				String hiddenLayerString) {
			super();
			this.learningRate = learningRate;
			this.momentum = momentum;
			this.trainingEpochs = trainingEpochs;
			this.validationSetSize = validationSetSize;
			this.hiddenLayerString = hiddenLayerString;
		}

		public float getLearningRate() {
			return learningRate;
		}

		public float getMomentum() {
			return momentum;
		}

		public int getTrainingEpochs() {
			return trainingEpochs;
		}

		public int getValidationSetSize() {
			return validationSetSize;
		}

		public String getHiddenLayerString() {
			return hiddenLayerString;
		}
		
		public String getComputedHiddenLayerString(Instances trainingInstances) {
			if (hiddenLayerString.equals("a"))
				return hiddenLayerString;
			else {
				int numPerLayer = (trainingInstances.numAttributes() + trainingInstances.numClasses()) / 4;
				return numPerLayer + "," + numPerLayer;
			}
		}
		
	}
	

}
