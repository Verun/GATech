package com.gatech.cs7641.assignment1.trainingRunner;

import java.util.List;

public interface TrainingRunner {

	List<SingleRunResult> runTraining();

	String getDescriptor();
}
