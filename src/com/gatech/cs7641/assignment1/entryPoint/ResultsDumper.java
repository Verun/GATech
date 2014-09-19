package com.gatech.cs7641.assignment1.entryPoint;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;

public class ResultsDumper {

	public String getResults(Iterable<SingleRunResult> results) {
		
		final String TAB = "\t";
		final String NEWLINE = "\n";
		
		StringBuilder sbr = new StringBuilder();
		
		
		
		for (SingleRunResult srr : results) {
			
			//print info about the relation
			AttributeSelectedInstances asi = srr.getAttributeSelectedInstances();
			
			String relName = asi.getAttributeSelectedInstances().relationName();
			String evaluator = asi.getEvaluatorDescriptorString();
			String searcher = asi.getSearcherDescriptorString();
			String keptIndices = join(asi.getAttributeIndicesKeptFromOriginalInstance());
			
			sbr.append(relName);
			sbr.append(TAB);
			sbr.append(evaluator);
			sbr.append(TAB);
			sbr.append(searcher);
			sbr.append(TAB);
			sbr.append(keptIndices);
			sbr.append(TAB);
			
			//info abt the classifier used
			ClassifierWithDescriptor cwd = srr.getClassifierWithDescriptor();
			sbr.append(cwd.getDescriptor());
			sbr.append(TAB);
			
			//training set size
			sbr.append(cwd.getInstancesTrainedOn().numInstances());
			sbr.append(TAB);
			
			//time it took to train
			sbr.append(cwd.getTimeItTookToTrain());
			sbr.append(TAB);
			
			//training stats (error rates etc)
			
			//test stats (error rates etc)
			
			
			
			sbr.append(NEWLINE);
		}
		
		return sbr.toString();
	}
	
	private String join(int[] arr) {
		StringBuilder sbr = new StringBuilder();
		
		for (int i : arr) {
			if (sbr.length() > 0)
				sbr.append(",");
			sbr.append(i);
		}
		
		return sbr.toString();
	}
}
