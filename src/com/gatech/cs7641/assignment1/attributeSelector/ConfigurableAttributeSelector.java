package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.Iterator;
import java.util.List;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;

public class ConfigurableAttributeSelector implements AttributeSelector {

	private final Iterable<ASEvaluation> evaluators;
	private final Iterable<ASSearch> searchers;
	
	public ConfigurableAttributeSelector(Iterable<ASEvaluation> evaluators,
			Iterable<ASSearch> searchers) {
		super();
		this.evaluators = evaluators;
		this.searchers = searchers;
	}

	public Iterable<InstancesWithSelectedIndices> getAttributeSelectedInstances(
			final Instances original) {
		
		return new Iterable<InstancesWithSelectedIndices>() {

			@Override
			public Iterator<InstancesWithSelectedIndices> iterator() {

				return new AbstractIterator<InstancesWithSelectedIndices>() {

					private final List<ASEvaluation> evals = Lists.newArrayList(evaluators); 
					private final List<ASSearch> srchs = Lists.newArrayList(searchers);
					private int evalIndex = 0;
					private int searchIndex = 0;
					
					@Override
					protected InstancesWithSelectedIndices computeNext() {

						if (searchIndex >= srchs.size()) {
							searchIndex = 0;
							evalIndex++;
						}
						
						if (evalIndex >= evals.size())
							return endOfData();
						
						ASEvaluation evalToUse = evals.get(evalIndex);
						ASSearch searcherToUse = srchs.get(searchIndex);
						
						searchIndex++;
						
						Instances copy = new Instances(original, 0, original.numInstances());
						
						AttributeSelection attrSelection = new AttributeSelection();
						attrSelection.setEvaluator(evalToUse);
						attrSelection.setSearch(searcherToUse);

						try {
							attrSelection.SelectAttributes(copy);
							
							int[] selectedIndices = attrSelection.selectedAttributes();
							
							return new InstancesWithSelectedIndices(attrSelection.reduceDimensionality(copy), selectedIndices, original, evalToUse.getClass().getName(), searcherToUse.getClass().getName());
							
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
							
							throw new RuntimeException(e);
						}
						

						
					}
					
				};
				
			}
			
		};
		
	}


}
