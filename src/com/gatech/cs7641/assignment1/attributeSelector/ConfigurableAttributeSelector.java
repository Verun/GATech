package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.core.Instances;
import weka.filters.Filter;
import weka.attributeSelection.AttributeSelection;

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

	public Iterable<AttributeSelectedInstances> getAttributeSelectedInstances(
			final Instances original) {
		
		return new Iterable<AttributeSelectedInstances>() {

			@Override
			public Iterator<AttributeSelectedInstances> iterator() {

				return new AbstractIterator<AttributeSelectedInstances>() {

					private final List<ASEvaluation> evals = Lists.newArrayList(evaluators); 
					private final List<ASSearch> srchs = Lists.newArrayList(searchers);
					private int evalIndex = 0;
					private int searchIndex = 0;
					private final Set<String> returned = new HashSet<String>();
					
					@Override
					protected AttributeSelectedInstances computeNext() {

						//keep searching for a unique set of attributes we haven't previously considered
						while (true) {
							try {					
								if (searchIndex >= srchs.size()) {
									searchIndex = 0;
									evalIndex++;
								}
								
								if (evalIndex >= evals.size())
									return endOfData();
								
								ASEvaluation evalToUse = evals.get(evalIndex);
								ASSearch searcherToUse = srchs.get(searchIndex);
								
								searchIndex++;
								
								//Instances copy = new Instances(original, 0, original.numInstances());
								//copy.setClassIndex(original.classIndex());
								
								AttributeSelection attrSelection = new AttributeSelection();
								attrSelection.setEvaluator(evalToUse);
								attrSelection.setSearch(searcherToUse);

								attrSelection.SelectAttributes(original);
								
								String hashedSelectedIndices = hashSelectedIndices(attrSelection.selectedAttributes());
								
								if (returned.contains(hashedSelectedIndices)) {
									System.out.println("Ignoring search: " +  searcherToUse.getClass().getName());
									continue;
								} else
									returned.add(hashedSelectedIndices);
								
//								attrSelection.setInputFormat(original);
								
//								Instances attributeSelected = Filter.useFilter(original, attrSelection);
//
//								if (attributeSelected.numAttributes() == original.numAttributes())
//									continue; //no attributes selected
//								
//								int[] selectedIndicesFromOriginal = getSelectedIndicesFromOriginal(original, attributeSelected);	
//								
								System.out.println("hashed: " + hashedSelectedIndices);
//								
								Instances reduced = attrSelection.reduceDimensionality(original);
								
								System.out.println("index of class attr in reduced: " + reduced.classIndex());
								
								return new AttributeSelectedInstances(reduced, attrSelection.selectedAttributes(), original, evalToUse.getClass().getName(), searcherToUse.getClass().getName());
								
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
								
								throw new RuntimeException(e);
							}

						}//end while
					}
					
					private String hashSelectedIndices(int[] selectedIndices) {
						int[] copy = Arrays.copyOf(selectedIndices, selectedIndices.length);
						Arrays.sort(copy);
						
						StringBuilder sbr = new StringBuilder();
						
						for (int i : copy) {
							sbr.append(i);
							sbr.append("_");
						}
						
						return sbr.toString();
					}
					
				};
				
			}
			
		};
		
	}

}
