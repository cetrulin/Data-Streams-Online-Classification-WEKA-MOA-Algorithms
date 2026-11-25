/*
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package moa.classifiers.igngsvm;

import java.lang.reflect.Field;
import java.util.ArrayList;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.igngsvm.gng.GUnit;
import moa.classifiers.igngsvm.gng.GNG;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.WEKAClassOption;
import weka.classifiers.*;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.WilsonEditing;

/**
 * Incremental Growing Neural Gas Support Vector Machine (iGNGSVM) for MOA.
 *
 * <p>Combines per-class Growing Neural Gas ({@link GNG}) prototype reduction,
 * optional SMOTE class balancing, Support Vector inheritance, Wilson's Edited
 * Nearest-Neighbor ({@link WilsonEditing}) noise filtering, and a LibSVM
 * classifier trained incrementally over data blocks.</p>
 *
 * @author Andres L. Suarez-Cetrulo
 */
public class IGNGSVM extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	// IGNGSVM block parameters
	public IntOption tsOption = new IntOption("blocksSize", 't', "BlockSize", 10100);
	public IntOption modoTsOption = new IntOption("blocksMode", 'T', "BlockMode", 0);

	// GNG parameters
	public IntOption lambdaOption = new IntOption("lambda", 'l', "Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm', "MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", 'a', "Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", 'd', "d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", 'Z', "EpsilonB", 0.2);
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'K', "EpsilonN", 0.006);
	public IntOption stoppingCriteriaOption = new IntOption("stoppingCriteria", 'c', "Stopping criteria", 0);
	public FloatOption perParadaOption = new FloatOption("stoppingMode", 'Q', "Stopping mode percentage (0 uses fixed stoppingCriteria)", 20);
	public IntOption modoHerenciaOption = new IntOption("svInheritance", 'e', "SV inheritance mode (0: iGNGSVM1, 1: iGNGSVM2, >1: none)", 0);
	public IntOption smoteNNOption = new IntOption("smoteNeighbors", 'X', "Number of nearest neighbors in SMOTE (0 disables SMOTE)", 5);

	// LibSVM base learner option
	public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'b',
			"Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.functions.LibSVM");

	// Wilson Editing filter parameters
	public IntOption m_kNNOption = new IntOption("wilsonNeighbors", 'n', "Number of neighbors for Wilson Editing", 3);
	public IntOption vecesWilsonOption = new IntOption("wilsonIterations", 'V', "Number of Wilson Editing passes per block", 1);
	public IntOption wilsonModeOption = new IntOption("wilsonMode", 'M', "Wilson mode (0: normal, 1: SVs subpopulation only)", 0);

	// SVM model
	public LibSVM svm;

	// Prototype sets per class
	private ArrayList<ArrayList<GUnit>> S;

	// Support Vector set
	public ArrayList<GUnit> SV;

	protected int numberInstances;

	// Training block TS
	protected Instances TS;

	// Per-class subsets of TS
	public ArrayList<Instances> TSi;

	protected Instances instancesBuffer;

	protected boolean isClassificationEnabled;

	protected boolean isBufferStoring;

	protected int valorBufferTS;

	protected boolean trainTS = false;

	// Nearest-neighbor search
	protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

	@Override
	public boolean isRandomizable() {
		return false;
	}

	boolean m_debug = false;
	boolean m_exportS = false;
	boolean m_export = true;

	@Override
	public void resetLearningImpl() {
		SV = new ArrayList<GUnit>();
		S = new ArrayList<ArrayList<GUnit>>();

		try {
			String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
			createWekaClassifier(options);
		} catch (Exception e) {
			System.err.println("Creating a new classifier: " + e.getMessage());
		}

		numberInstances = 0;
		isClassificationEnabled = false;
		this.isBufferStoring = true;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		// Auxiliary variables
		double[][] SVpnew = null;
		int[] label, nSV, svClass;
		int previous;

		Instances Temp;

		// Per-class GNG list
		ArrayList<GNG> GNGi = new ArrayList<GNG>();
		ArrayList<Double> labels = new ArrayList<Double>();
		ArrayList<Instances> Splus = new ArrayList<Instances>();

		try {
			if (numberInstances == 0) {
				this.instancesBuffer = new Instances(((Instance) (inst.copy())).dataset());
				this.instancesBuffer.clear();
				this.valorBufferTS = inst.index(0);

				if (svm instanceof UpdateableClassifier) {
					svm.buildClassifier(instancesBuffer);
					this.isClassificationEnabled = true;
				} else {
					this.isBufferStoring = true;
				}
			}
			numberInstances++;

			if (svm instanceof UpdateableClassifier) {
				if (numberInstances > 0) {
					((UpdateableClassifier) svm).updateClassifier(inst);
				}
			} else {

				if (isBufferStoring == true) {
					instancesBuffer.add(inst);
					if (modoTsOption.getValue() != 0 && (instancesBuffer.get(instancesBuffer.numInstances() - 1).value(0) != valorBufferTS)) {
						valorBufferTS = (int) instancesBuffer.get(instancesBuffer.numInstances() - 1).value(0);
						if (instancesBuffer.size() > 1) {
							trainTS = true;
						}
					}
				}

				// Trigger block training based on block size or first-attribute boundary
				if (modoTsOption.getValue() == 0 && (numberInstances % tsOption.getValue() == 0 && instancesBuffer.size() == tsOption.getValue())) {
					trainTS = true;
				}
				if (trainTS) {
					trainTS = false;
					// Build TS from buffered instances
					TS = new Instances(instancesBuffer);
					TSi = new ArrayList<Instances>();
					isBufferStoring = false;
					S.clear();

					// Partition block TS into per-class subsets TSi
					for (int i = 0; i < TS.numInstances(); i++) {
						Instance aux = (Instance) TS.get(i).copy();
						double auxLabel = aux.classValue();

						if (labels.contains(auxLabel)) {
							Instances TsiAux = TSi.get(labels.indexOf(auxLabel));
							TsiAux.add(aux);
							TSi.set(labels.indexOf(auxLabel), TsiAux);
						} else {
							Instances TsiAux = new Instances(((Instance) (inst.copy())).dataset());
							TsiAux.clear();
							TsiAux.add(aux);
							TSi.add(TsiAux);
							labels.add(auxLabel);
						}
					}

					// SV INHERITANCE MODE 2 (merge previous SVs into TSi before GNG)
					Instances SVs = new Instances(inst.dataset());
					SVs.clear();

					if (modoHerenciaOption.getValue() == 1) {
						for (int i = 0; i < SV.size(); i++) {
							Instance instSV = (Instance) inst.copy();
							for (int j = 0; j < SV.get(i).w.length; j++) {
								instSV.setValue(j, SV.get(i).w[j]);
							}
							instSV.setClassValue(SV.get(i).getLabel());
							SVs.add(instSV);
						}

						// Partition SVs into per-class subsets
						for (int i = 0; i < SVs.numInstances(); i++) {
							Instance aux = (Instance) SVs.get(i).copy();
							double auxLabel = aux.classValue();

							if (labels.contains(auxLabel)) {
								Instances TsiAux = TSi.get(labels.indexOf(auxLabel));
								TsiAux.add(aux);
								TSi.set(labels.indexOf(auxLabel), TsiAux);
							} else {
								Instances TsiAux = new Instances(((Instance) (inst.copy())).dataset());
								TsiAux.clear();
								TsiAux.add(aux);
								TSi.add(TsiAux);
								labels.add(auxLabel);
							}
						}
					}

					// START GNG: train one GNG topology per class subset
					for (int i = 0; i < TSi.size(); i++) {
						GNG aux = new GNG();
						aux.resetLearningImpl();

						aux.lambdaOption = this.lambdaOption;
						aux.alfaOption = this.alfaOption;
						aux.maxAgeOption = this.maxAgeOption;
						aux.constantOption = this.constantOption;
						aux.BepsilonOption = this.BepsilonOption;
						aux.NepsilonOption = this.NepsilonOption;

						// Set stopping criterion (percentage of TSi or fixed count)
						if (perParadaOption.getValue() != 0) {
							IntOption stopTSi = (IntOption) stoppingCriteriaOption.copy();
							int percentStop = (int) ((perParadaOption.getValue() * (double) TSi.get(i).size()) / (double) 100);
							if (percentStop >= 1) {
								stopTSi.setValue(percentStop);
								aux.stoppingCriteriaOption = stopTSi;
								if (m_debug) {
									System.out.println("GNG stops on this TSi at " + stopTSi.getValue() + " generated prototypes");
								}
							} else {
								aux.stoppingCriteriaOption = this.stoppingCriteriaOption;
							}
						} else {
							aux.stoppingCriteriaOption = this.stoppingCriteriaOption;
						}

						// Feed instances from class subset i until stopping criterion is met
						if (TSi.get(i).numInstances() == 1) {
							aux.trainOnInstanceImpl(TSi.get(i).get(0));
						} else if (TSi.get(i).numInstances() > 1) {
							for (int j = 0; aux.getNumberOfPrototypesCreated() < aux.stoppingCriteriaOption.getValue(); j++) {
								aux.trainOnInstanceImpl(TSi.get(i).get(j));
								if (j + 1 == TSi.get(i).numInstances()) {
									j = -1;
								}
							}
						}
						GNGi.add(aux);
						S.add(GNGi.get(i).getS());
					}

					if (m_exportS) {
						System.out.println();
						System.out.println("PROTOTYPE LIST S:");
						System.out.println();
						for (int i = 0; i < S.size(); i++) {
							for (int j = 0; j < S.get(i).size(); j++) {
								System.out.println(S.get(i).get(j) + ", " + labels.get(i));
							}
						}
					}

					Temp = new Instances(inst.dataset());
					SVpnew = null;

					// Combine all per-class prototype sets Si into Splus
					int counter = 0;
					for (int i = 0; i < S.size(); i++) {
						Temp.clear();
						for (int k = 0; k < S.get(i).size(); k++, counter++) {
							Instance instS = (Instance) inst.copy();
							for (int j = 0; j < S.get(i).get(k).w.length; j++) {
								instS.setValue(j, S.get(i).get(k).w[j]);
							}
							instS.setClassValue(labels.get(i));
							Temp.add(instS);
						}
						Splus.add(new Instances(Temp));
					}
					// END GNG

					// START SMOTE
					if (smoteNNOption.getValue() != 0) {
						int greatestLabel = 0;
						for (int i = 0; i < Splus.size(); i++) {
							if (Splus.get(i).numInstances() > greatestLabel) {
								greatestLabel = Splus.get(i).numInstances();
							}
						}
						for (int i = 0; i < Splus.size(); i++) {
							if (Splus.get(i).numInstances() != greatestLabel) {
								System.out.println("SIZE BEFORE SMOTE: " + Splus.get(i).numInstances());
								SMOTE filters = new SMOTE();
								filters.setInputFormat(Splus.get(i));
								filters.setNearestNeighbors(smoteNNOption.getValue());
								filters.setPercentage((double) (greatestLabel - Splus.get(i).numInstances()) / (double) Splus.get(i).numInstances() * 100);
								Splus.set(i, Filter.useFilter(Splus.get(i), filters));
								System.out.println("SIZE AFTER SMOTE: " + Splus.get(i).numInstances());
							}
						}
					}
					// END SMOTE

					Temp.clear();
					// Merge Splus into a single Instances container (Temp)
					for (int i = 0; i < Splus.size(); i++) {
						for (int j = 0; j < Splus.get(i).size(); j++) {
							Temp.add(Splus.get(i).get(j));
						}
					}

					// SV INHERITANCE MODE 1: combine S* with SVs from the previous iteration
					if (modoHerenciaOption.getValue() == 0) {
						SVs = new Instances(Temp);
						SVs.clear();
						for (int i = counter; i < SV.size() + counter; i++) {
							Instance instSV = (Instance) inst.copy();
							for (int j = 0; j < SV.get(i - counter).w.length; j++) {
								instSV.setValue(j, SV.get(i - counter).w[j]);
							}
							instSV.setClassValue(SV.get(i - counter).getLabel());
							if (wilsonModeOption.getValue() == 0) {
								Temp.add(instSV);
							} else {
								SVs.add(instSV);
							}
						}
					}

					// START WILSON EDITING
					for (int wilsonPass = 0; wilsonPass < vecesWilsonOption.getValue(); wilsonPass++) {
						System.out.println("SIZE BEFORE WILSON it. " + wilsonPass + ": " + Temp.numInstances());
						WilsonEditing filterW = new WilsonEditing();
						filterW.setInputFormat(Temp);
						filterW.setNearestNeighbors(m_kNNOption.getValue());
						filterW.setWilsonMode(wilsonModeOption.getValue());
						if (wilsonModeOption.getValue() == 1) {
							filterW.setWilsonSubPopulation(SVs);
						}
						Temp = Filter.useFilter(Temp, filterW);
						System.out.println("SIZE AFTER WILSON it. " + wilsonPass + ": " + Temp.numInstances());
					}
					// END WILSON EDITING

					for (int i = 0; i < Temp.numInstances(); i++) {
						System.out.println(Temp.get(i));
					}

					try {
						// Train a new SVM with the filtered set Temp
						buildClassifier(Temp);
						System.out.println("Successfully built GNGSVM classifier");
						System.out.println("instancesBuffer length: " + instancesBuffer.numInstances());
						isClassificationEnabled = true;
						instancesBuffer = new Instances(((Instance) (inst.copy())).dataset());
						instancesBuffer.clear();
						isBufferStoring = true;
					} catch (Exception e) {
						e.printStackTrace();
					}

					// Extract the new Support Vector set SVpnew
					label = (int[]) getField(svm.m_Model, "label");
					nSV = (int[]) getField(svm.m_Model, "nSV");
					Object[][] o = (Object[][]) getField(svm.m_Model, "SV");
					SVpnew = new double[o.length][];
					svClass = new int[SVpnew.length];

					for (int i = 0; i < o.length; i++) {
						SVpnew[i] = new double[o[i].length];
						for (int j = 0; j < o[i].length; j++) {
							SVpnew[i][j] = (Double) getField(o[i][j], "value");
						}
						previous = 0;
						for (int j = 0; j < nSV.length;) {
							if (i < nSV[j] + previous) {
								svClass[i] = label[j];
								j = nSV.length;
							} else {
								previous += nSV[j];
								j++;
							}
						}
					}

					// Update retained Support Vector set SV
					SV.clear();
					for (int i = 0; i < SVpnew.length; i++) {
						SV.add(new GUnit(SVpnew[i], svClass[i]));
					}

					if (m_export) {
						System.out.println();
						System.out.println("SUPPORT VECTOR LIST:");
						System.out.println();
					}
					for (int i = 0; i < SV.size(); i++) {
						System.out.println(SV.get(i) + "," + svClass[i]);
					}

					if (m_export) {
						System.out.println();
						System.out.println("End of block TS");
						System.out.println();
						System.out.println("//////////////////////////////////////");
					}
				}
			}

		} catch (Exception e) {
			System.err.println("Training: " + e.getMessage());
			e.printStackTrace();
		}
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] votes = new double[inst.numClasses()];

		if (isClassificationEnabled == false) {
			for (int i = 0; i < inst.numClasses(); i++) {
				votes[i] = 1.0 / inst.numClasses();
			}
		} else {
			try {
				votes = svm.distributionForInstance(inst);
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		return votes;
	}

	public void buildClassifier(Instances Temp) {
		try {
			if ((svm instanceof UpdateableClassifier) == false) {
				try {
					svm = new LibSVM();
					String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
					createWekaClassifier(options);
					svm.buildClassifier(Temp);
				} catch (Exception e) {
					System.err.println("Creating a new classifier: " + e.getMessage());
				}
				isBufferStoring = false;
			}
		} catch (Exception e) {
			System.err.println("Building WEKA Classifier: " + e.getMessage());
		}
	}

	public void createWekaClassifier(String[] options) throws Exception {
		String classifierName = options[0];
		String[] newoptions = options.clone();
		newoptions[0] = "";
		this.svm = (LibSVM) weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);
	}

	/**
	 * Returns the current value of the specified public field via reflection.
	 *
	 * @param o    the object the field belongs to
	 * @param name the name of the field
	 * @return the field value, or {@code null} on error
	 */
	protected Object getField(Object o, String name) {
		Field f;
		Object result;

		try {
			f = o.getClass().getField(name);
			result = f.get(o);
		} catch (Exception e) {
			e.printStackTrace();
			result = null;
		}

		return result;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("iGNGSVM v2 by Andres L. Suarez-Cetrulo. " + svm.toString());
	}

	@Override
	protected moa.core.Measurement[] getModelMeasurementsImpl() {
		return null;
	}
}
