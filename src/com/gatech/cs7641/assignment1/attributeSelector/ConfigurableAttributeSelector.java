package com.gatech.cs7641.assignment1.attributeSelector;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;

public class ConfigurableAttributeSelector implements AttributeSelector {

	private final Iterable<ASEvaluation> evaluators;
	private final Iterable<ASSearch> searchers;
	private final boolean returnAllAttributesTheFirstTime;

	public ConfigurableAttributeSelector(
			final Iterable<ASEvaluation> evaluators,
			final Iterable<ASSearch> searchers,
			final boolean returnAllAttributesTheFirstTime) {
		super();
		this.evaluators = evaluators;
		this.searchers = searchers;
		this.returnAllAttributesTheFirstTime = returnAllAttributesTheFirstTime;
	}

	@Override
	public Iterable<AttributeSelectedInstances> getAttributeSelectedInstances(
			final Instances original) {

		return new Iterable<AttributeSelectedInstances>() {

			@Override
			public Iterator<AttributeSelectedInstances> iterator() {

				return new AbstractIterator<AttributeSelectedInstances>() {

					private final List<ASEvaluation> evals = Lists
							.newArrayList(evaluators);
					private final List<ASSearch> srchs = Lists
							.newArrayList(searchers);
					private int evalIndex = 0;
					private int searchIndex = 0;
					private final Set<String> returned = new HashSet<String>();
					private boolean returnedAllAttributes;

					@Override
					protected AttributeSelectedInstances computeNext() {

						if (returnAllAttributesTheFirstTime
								&& !returnedAllAttributes) {
							returnedAllAttributes = true;
							return new AttributeSelectedInstances(original,
									getArrayOfAttributeIndices(original),
									original, "N/A", "N/A");
						}

						// keep searching for a unique set of attributes we
						// haven't previously considered
						while (true) {
							try {
								if (searchIndex >= srchs.size()) {
									searchIndex = 0;
									evalIndex++;
								}

								if (evalIndex >= evals.size())
									return endOfData();

								final ASEvaluation evalToUse = evals
										.get(evalIndex);
								final ASSearch searcherToUse = srchs
										.get(searchIndex);

								searchIndex++;

								// Instances copy = new Instances(original, 0,
								// original.numInstances());
								// copy.setClassIndex(original.classIndex());

								final AttributeSelection attrSelection = new AttributeSelection();
								attrSelection.setEvaluator(evalToUse);
								attrSelection.setSearch(searcherToUse);

								attrSelection.SelectAttributes(original);

								final String hashedSelectedIndices = hashSelectedIndices(attrSelection
										.selectedAttributes());

								if (returned.contains(hashedSelectedIndices)) {
									System.out
											.println("Ignoring search: "
													+ searcherToUse.getClass()
															.getName()
													+ " because indices have been explored before: "
													+ hashedSelectedIndices);
									continue;
								} else
									returned.add(hashedSelectedIndices);

								// attrSelection.setInputFormat(original);

								// Instances attributeSelected =
								// Filter.useFilter(original, attrSelection);
								//
								// if (attributeSelected.numAttributes() ==
								// original.numAttributes())
								// continue; //no attributes selected
								//
								// int[] selectedIndicesFromOriginal =
								// getSelectedIndicesFromOriginal(original,
								// attributeSelected);
								//
								System.out.println("hashed: "
										+ hashedSelectedIndices);
								//
								final Instances reduced = attrSelection
										.reduceDimensionality(original);

								System.out
										.println("index of class attr in reduced: "
												+ reduced.classIndex());

								return new AttributeSelectedInstances(reduced,
										attrSelection.selectedAttributes(),
										original, evalToUse.getClass()
												.getName(), searcherToUse
												.getClass().getName());

							} catch (final Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();

								throw new RuntimeException(e);
							}

						}// end while
					}

					private String hashSelectedIndices(
							final int[] selectedIndices) {
						final int[] copy = Arrays.copyOf(selectedIndices,
								selectedIndices.length);
						Arrays.sort(copy);

						final StringBuilder sbr = new StringBuilder();

						for (final int i : copy) {
							sbr.append(i);
							sbr.append("_");
						}

						return sbr.toString();
					}

					private int[] getArrayOfAttributeIndices(
							final Instances instances) {
						final int numAttributes = instances.numAttributes();

						final int[] toReturn = new int[numAttributes];
						for (int x = 0; x < numAttributes; x++) {
							toReturn[x] = x;
						}

						return toReturn;
					}

				};

			}

		};

	}

}
