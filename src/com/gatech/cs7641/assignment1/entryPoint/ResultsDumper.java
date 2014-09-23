package com.gatech.cs7641.assignment1.entryPoint;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import com.gatech.cs7641.assignment1.attributeSelector.AttributeSelectedInstances;
import com.gatech.cs7641.assignment1.trainingRunner.ClassifierWithDescriptor;
import com.gatech.cs7641.assignment1.trainingRunner.SingleRunResult;

public class ResultsDumper {

	public void dumpResultsToFile(final Iterable<SingleRunResult> results,
			final String fileName) throws FileNotFoundException, IOException {

		final File outFile = new File(fileName);
		if (outFile.exists())
			outFile.delete();

		final String TAB = "\t";
		final String NEWLINE = "\n";

		final StringBuilder sbr = new StringBuilder();

		try (

		FileOutputStream fos = new FileOutputStream(outFile, false);
				OutputStreamWriter srw = new OutputStreamWriter(fos);
				BufferedWriter brw = new BufferedWriter(srw);

		) {

			for (final SingleRunResult srr : results) {

				sbr.setLength(0);

				// print info about the relation
				final AttributeSelectedInstances asi = srr
						.getAttributeSelectedInstances();

				final String relName = asi
						.getAttributeSelectedInstances()
						.relationName()
						.substring(
								0,
								asi.getAttributeSelectedInstances()
										.relationName().indexOf("-"));
				final String evaluator = asi.getEvaluatorDescriptor();
				final String searcher = asi.getSearcherDescriptor();
				final String keptIndices = asi
						.getAttributeIndicesKeptFromOriginalInstance().length == asi
						.getInstancesBeforeAttributeSelection().numAttributes() ? "all"
						: join(asi
								.getAttributeIndicesKeptFromOriginalInstance());

				sbr.append(relName);
				sbr.append(TAB);
				sbr.append(evaluator);
				sbr.append(TAB);
				sbr.append(searcher);
				sbr.append(TAB);
				sbr.append(keptIndices);
				sbr.append(TAB);

				// info abt the classifier used
				final ClassifierWithDescriptor cwd = srr
						.getClassifierWithDescriptor();
				sbr.append(cwd.getDescriptor());
				sbr.append(TAB);

				// training set size
				sbr.append(cwd.getInstancesTrainedOn().numInstances());
				sbr.append(TAB);

				// time it took to train (in seconds)
				sbr.append(String.format("%.3f",
						cwd.getTimeItTookToTrain() / 1000.00d));
				sbr.append(TAB);

				// training stats (error rates etc)
				sbr.append(String.format("%.3f", srr.getTrainingEvaluation()
						.weightedPrecision()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTrainingEvaluation()
						.weightedRecall()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTrainingEvaluation()
						.pctCorrect()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTrainingEvaluation()
						.pctIncorrect()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTrainingEvaluation()
						.pctUnclassified()));
				sbr.append(TAB);

				// test stats (error rates etc)
				sbr.append(String.format("%.3f", srr.getTestEvaluation()
						.weightedPrecision()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTestEvaluation()
						.weightedRecall()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTestEvaluation()
						.pctCorrect()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTestEvaluation()
						.pctIncorrect()));
				sbr.append(TAB);
				sbr.append(String.format("%.3f", srr.getTestEvaluation()
						.pctUnclassified()));
				sbr.append(TAB);

				brw.write(sbr.toString());
				brw.newLine();
			}

		}

		// return sbr.toString();
	}

	private String join(final int[] arr) {
		final StringBuilder sbr = new StringBuilder();

		for (final int i : arr) {
			if (sbr.length() > 0)
				sbr.append(",");
			sbr.append(i);
		}

		return sbr.toString();
	}
}
