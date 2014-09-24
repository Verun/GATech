package com.gatech.cs7641.assignment1.trainingRunner.svm;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelector;
import com.gatech.cs7641.assignment1.datasetPartitioner.DatasetPartitioner;
import com.gatech.cs7641.assignment1.entryPoint.GlobalConstants;
import com.gatech.cs7641.assignment1.trainingRunner.BaseTrainingRunner;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.google.common.collect.AbstractIterator;

public class SVMTrainingRunner extends BaseTrainingRunner {

	public SVMTrainingRunner(AttributeSelector attrSelector,
			DatasetPartitioner partitioner, Instances originalTrainingSet,
			Instances testSet) {
		super(attrSelector, partitioner, originalTrainingSet, testSet);
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getDescriptor() {
		return "SVM";
	}

	@Override
	protected Iterable<ClassifierWithDescriptor> buildClassifiers(
			final Instances trainingInstances) {

		return new Iterable<ClassifierWithDescriptor>() {

			@Override
			public Iterator<ClassifierWithDescriptor> iterator() {

				return new AbstractIterator<ClassifierWithDescriptor>() {

						private final List<LibSVM> libSvms = new ArrayList<LibSVM>();
						private final List<String> descriptors = new ArrayList<String>();
						private int indexIntoSvms = 0;
						
						{
							final double CACHE_SIZE = 100;
							
							for (double cost : new double[] {Math.pow(2.0f, -5f), Math.pow(2.0, -3), Math.pow(2.0, -1), Math.pow(2.0, 1), Math.pow(2.0, 3), Math.pow(2.0, 5)})
								for (int degree : new int[] {1,2,3,4,5})
									for (int kernelTag : new int[] {LibSVM.KERNELTYPE_LINEAR, LibSVM.KERNELTYPE_RBF})
									{
										if (kernelTag == LibSVM.KERNELTYPE_LINEAR) {
											
											LibSVM libSvm = new LibSVM();
											libSvm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
											libSvm.setCacheSize(CACHE_SIZE);
											libSvm.setCost(cost);
											libSvm.setDegree(degree);
											
											SelectedTag st = new SelectedTag(kernelTag, LibSVM.TAGS_KERNELTYPE);
											libSvm.setKernelType(st);
											
											libSvm.setSeed(GlobalConstants.RAND_SEED);
											libSvms.add(libSvm);
											
											descriptors.add("c="+cost+";d="+degree+";k="+st.getSelectedTag().getIDStr());
											
											
										} else {
											
											for (double gamma : new double[] {Math.pow(2.0f, -5f), Math.pow(2.0, -3), Math.pow(2.0, -1), Math.pow(2.0, 1), Math.pow(2.0, 3), Math.pow(2.0, 5)}) {
											
												LibSVM libSvm = new LibSVM();
												libSvm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
												libSvm.setCacheSize(CACHE_SIZE);
												libSvm.setCost(cost);
												libSvm.setDegree(degree);
												libSvm.setGamma(gamma);
												
												SelectedTag st = new SelectedTag(kernelTag, LibSVM.TAGS_KERNELTYPE);
												libSvm.setKernelType(st);
												
												libSvm.setSeed(GlobalConstants.RAND_SEED);
												libSvms.add(libSvm);
											
												descriptors.add("c="+cost+";d="+degree+";k="+st.getSelectedTag().getIDStr()+";g="+gamma);
											
											}
											
												
											
										}
										
									}

						}
					
					
						@Override
						protected ClassifierWithDescriptor computeNext() {

							if (indexIntoSvms >= libSvms.size())
								return endOfData();
							
							long trainingTime;
							String descriptor = descriptors.get(indexIntoSvms);
							LibSVM libSvm = libSvms.get(indexIntoSvms);
							indexIntoSvms++;
							try {
								final long start = System.currentTimeMillis();
								libSvm.buildClassifier(trainingInstances);
								trainingTime = System.currentTimeMillis() - start;
							} catch (final Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();

								throw new RuntimeException(e);
							}

							return new ClassifierWithDescriptor(libSvm, descriptor,
									trainingInstances, trainingTime);
							
						}

						
						
						
					};

				};

			};

		};

	};

