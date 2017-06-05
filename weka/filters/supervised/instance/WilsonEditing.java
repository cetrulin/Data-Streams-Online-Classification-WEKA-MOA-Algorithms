/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * WilsonEditing.java
 * 
 * Copyright (C) 2014 Andres L. Suarez Cetrulo
 * Copyright (C) 2014 Universidad Carlos III, Madrid, Spain
 */

package weka.filters.supervised.instance;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import moa.options.IntOption;

/**
 <!-- globalinfo-start -->
 * Undersamples a dataset by applying pruning with the Asymptotic Properties of Nearest Neighbor Rules Using Edited Data. For more information, see <br/>
 * <br/>
 * Dennis L. Wilson (1972). Synthetic Minority Over-sampling Technique. IEEE Transactions on Systems, Man, and Cybernetics. 16:321-357.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Wilson.1972,
 *    author = {Dennis L. Wilson},
 *    journal = {IEEE Transactions on Systems, Man, and Cybernetics},
 *    pages = {408-421},
 *    title = {Synthetic Minority Over-sampling Technique},
 *    volume = {SMC-2},
 *    number = {3},
 *    year = {1972}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -M &lt;m_kNN&gt;
 *  Specifies the number of nearest neighbors to use.
 *  (default 3)</pre>
 * 
 * <pre> -J &lt;wilsonMode&gt;
 *  Specifies the WilsonEditing mode to use ('entire population'=>0 or 'sub-population'=>1).
 *  (default 0)
 * </pre>
 * 
 * 
 <!-- options-end -->
 *  
 * @author Andres L. Suarez-Cetrulo (suarezcetrulo@gmail.com)
 * @version $Revision: 0001 $
 */
public class WilsonEditing
  extends Filter 
  implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

  /** for serialization. */
  static final long serialVersionUID = 0001;

  /** debug mode **/
  protected boolean m_debug = false;
  
  /** the number of neighbors to use. */
  protected int m_NearestNeighbors = 3;
    
  /** the Wilson Editing Mode. 1=>m_subPopulationMode=true */
  protected boolean m_subPopulationMode = false;
  protected Instances m_subpopulation = null;
  
  /**
   * Returns a string describing this classifier.
   * 
   * @return 		a description of the classifier suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "" + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return 		the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);

    result.setValue(Field.AUTHOR, "Dennis L. Wilson");
    result.setValue(Field.TITLE, "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data");
    result.setValue(Field.JOURNAL, "IEEE Transactions on Systems, Man, and Cybernetics");
    result.setValue(Field.YEAR, "1972");
    result.setValue(Field.VOLUME, "SMC-2");
    result.setValue(Field.NUMBER, "3");
    result.setValue(Field.PAGES, "408-421");

    return result;
  }

  /**
   * Returns the revision string.
   * 
   * @return 		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 0001 $");
  }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector();
    
    newVector.addElement(new Option(
	"\tSpecifies the number of nearest neighbors to use\n"
	+ "\t(default 3)",
	"M", 1, "-M <m_kNN>"));
    
    newVector.addElement(new Option(
	"\tSpecifies the number of consecutives WilsonEditing iterations to apply over the incoming dataset.\n"
	+ "\t(default 1)\n",
	"V", 1, "-V <percentage>"));
   
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -M &lt;m_kNN&gt;
   *  Specifies the number of nearest neighbors to use.
   *  (default 3)</pre>
   * 
   * <pre> -J &lt;wilsonMode&gt;
   *  Specifies the WilsonEditing mode to use ('entire population'=>0 or 'sub-population'=>1).
   *  (default 0)
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
	  	  
    String m_kNN = Utils.getOption('M', options);
    if (m_kNN.length() != 0) {
    	setNearestNeighbors(Integer.parseInt(m_kNN));
    } else {
    	setNearestNeighbors(3);
    }

    String wilsonMode = Utils.getOption('J', options);
    if (wilsonMode.length() != 0) {
      setWilsonMode(Integer.parseInt(wilsonMode));
    } else {
      setWilsonMode(0);
    } 

  }  


  /**
   * Gets the current settings of the filter.
   *
   * @return an array 	of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector<String>	result;
    
    result = new Vector<String>();
        
    result.add("-M");
    result.add("" + getNearestNeighbors());
    
    result.add("-J");
    result.add("" + getWilsonMode());
    
    return result.toArray(new String[result.size()]);
  }

  /**
   * Sets the Wilson Editing subpopulation (when Wilson Mode = 1)
   *  
   * @param value	the mode to use
   */
  public void setWilsonSubPopulation(Instances inst) {
    if (m_subPopulationMode){
      m_subPopulationMode = false;
      m_subpopulation = inst;
      
     } else
    	 System.out.println("The Wilson mode have to be equal to 1");  
  }

  /**
   * Gets the Wilson Editing subpopulation (when Wilson Mode = 1)
   *  
   * @param value	the mode to use
   */
  public Instances getWilsonSubPopulation(Instances inst) {
    if (m_subPopulationMode){
    	 return m_subpopulation;
    	 
     } else{
    	 System.out.println("The Wilson mode have to be equal to 1");  
    	 return null;
     }
  }
 
  
  /**
   * Sets the Wilson Editing Mode ('entire population'=>0 or 'sub-population'=>1).
   *  
   * @param value	the mode to use
   */
  public void setWilsonMode(int value) {
    if (value == 0)
      m_subPopulationMode = false;
    else if(value == 1)
      m_subPopulationMode = true;
    else
      System.err.println("Wilson Editing mode must be equal to 0 or 1 ");
  }
 
  
  
  /**
   * Gets the Wilson Editing Mode ('entire population'=>0 or 'sub-population'=>1).
   *  
   * @return 		the mode to use
   */
  public int getWilsonMode() {
	  if(m_subPopulationMode) 
		  return 1;
	  else 
		  return 0;
  }

  
  /**
   * Gets the Wilson Editing subpopulation if the mode is equal to 1
   *  
   * @return 		subpopulation
   */
  public Instances getSubPopulation() {
	  if(m_subPopulationMode) 
		  return m_subpopulation;
	  else 
		  return null;
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String nearestNeighborsTipText() {
    return "The number of nearest neighbors to use.";
  }

  /**
   * Sets the number of nearest neighbors to use.
   * 
   * @param value	the number of nearest neighbors to use
   */
  public void setNearestNeighbors(int value) {
    if (value >= 3)
      m_NearestNeighbors = value;
    else
      System.err.println("At least 3 neighbors necessary!");
  }

  /**
   * Gets the number of nearest neighbors to use.
   * 
   * @return 		the number of nearest neighbors to use
   */
  public int getNearestNeighbors() {
    return m_NearestNeighbors;
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  /*public String classValueTipText() {
    return "The index of the class value to which SMOTE should be applied. " +
    "Use a value of 0 to auto-detect the non-empty minority class.";
  }*/


  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo 	an Instances object containing the input 
   * 				instance structure (any instances contained in 
   * 				the object are ignored - only the structure is required).
   * @return 			true if the outputFormat may be collected immediately
   * @throws Exception 		if the input format can't be set successfully
   */
  public boolean setInputFormat(Instances instanceInfo) throws Exception {
    super.setInputFormat(instanceInfo);
    super.setOutputFormat(instanceInfo);
    return true;
  }

  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance 		the input instance
   * @return 			true if the filtered instance may now be
   * 				collected with output().
   * @throws IllegalStateException if no input structure has been defined
   */
  public boolean input(Instance instance) {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    if (m_FirstBatchDone) {
      push(instance);
      return true;
    } else {
      bufferInput(instance);
      return false;
    }
  }

  /**
   * Signify that this batch of input to the filter is finished. 
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return 		true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   * @throws Exception 	if provided options cannot be executed 
   * 			on input instances
   */
  public boolean batchFinished() throws Exception {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (!m_FirstBatchDone) {
      // Do Wilson, and clear the input instances.
      doWilson();
    }
    flushInput();

    m_NewBatch = true;
    m_FirstBatchDone = true;
    return (numPendingOutput() != 0);
  }
  
  /**
   * Select if one instance should be added into the reduced population
   * 
   * @param neighbours the neighbours of the targeted instance 
   * @param i 	targeted instance
   *
   * @return 		true if that instance is selected
   * */ 
  public boolean compareNeighbours(Instances neighbours, Instance i){	
	double halfOfKnn = (double) getNearestNeighbors()/2.0;  
	int numMatches = 0;
	
	for (int j = 0; j < neighbours.numInstances(); j++) {
		if(neighbours.get(j).classValue()==i.classValue()){
		    if(m_debug)
		    	System.out.println("NEIGHBOUR " + neighbours.get(j) + " MATCHES");
		    numMatches++;
		} if(numMatches > halfOfKnn){
		    if(m_debug)
		    	System.out.println("ADDED ON REDUCED DATASET => "+neighbours.get(j));
		    return true;
	    }				
	} return false;

  }
  
  /**
   * Prints information about selected options
   * */
  protected void printInfo(){
	  
	    System.out.println("Using " + getNearestNeighbors() + " neighbours");
	    
	    if(getWilsonMode()==0)
	    	System.out.println("Using Wilson Editing");
	    else
	    	System.out.println("Using Wilson Editing only under sub-population");	  
  }
  
  /**
   * The procedure implementing the Wilson Editing algorithm. The output
   * instances are pushed onto the output queue for collection.
   * 
   * @throws Exception 	if provided options cannot be executed 
   * 			on input instances
   */
  protected void doWilson() throws Exception {

	Instances tempAux, neighbours = null;
	Instances reduced = new Instances (getInputFormat()); //but it needs to be empty => reduced.clear() below
	Instance i = null; 
	Enumeration EnumData = null;
	printInfo(); //Prints information about selected options
	NearestNeighbourSearch m_NNSearch = new LinearNNSearch();
	
	if(m_debug) {
		System.out.println();
		System.out.println("Begin of WilsonEditing Filter");
		System.out.println();
		
	} if (m_subPopulationMode) { //wilsonMode=>1 only subpopulation							
		
		reduced.clear();
		EnumData = m_subpopulation.enumerateInstances();

		while (EnumData.hasMoreElements()) {
			//1 iterate over subpopulation
		    i = (Instance) EnumData.nextElement();
		    //2 iterate over dataset
		    tempAux = new Instances (getInputFormat());
		    //3 obtain neighbours
		    m_NNSearch.setInstances(tempAux);
		    //TODO: set distancias con kernel => m_NNSearch.getDistances();
		    neighbours = m_NNSearch.kNearestNeighbours(i, getNearestNeighbors());
		    //4 compare neighbours			    
		    if(compareNeighbours(neighbours, i))
		    	reduced.add(i);			    	

		} if(m_debug) {
			System.out.println(m_subpopulation.numInstances()+" SVs de la it. anterior");
			System.out.println(reduced.numInstances()+" SVs quedan");
		} 	

		
	} else { //wilsonMode=>0 the whole population
		
		reduced.clear();
		EnumData = getInputFormat().enumerateInstances();
	
		for (int count = 0;EnumData.hasMoreElements();count++) {
			//1 iterate over dataset
		    i = (Instance) EnumData.nextElement();
		    //2 avoid distances to himself 
		    tempAux = new Instances (getInputFormat());
		    tempAux.delete(count);
		    //3 obtain neighbours
		    m_NNSearch.setInstances(tempAux);
		    //TODO: set distancias con kernel => m_NNSearch.getDistances();
		    neighbours = m_NNSearch.kNearestNeighbours(i, getNearestNeighbors());
		    //4 compare neighbours			    
		    if(compareNeighbours(neighbours, i))
		    	reduced.add(i);	
		    
		} if(m_debug) {
			System.out.println(getInputFormat().numInstances()+" Prototipos de la it. anterior");
			System.out.println(reduced.numInstances()+" Prototipos quedan"); 
		} 

	} //return output
	if(m_subPopulationMode) {
		if(m_debug)
			System.out.println("Adding dataset in subpopulation mode");
		//subpopulation mode also returns the input dataset
		EnumData = getInputFormat().enumerateInstances();
		while(EnumData.hasMoreElements())
			push((Instance) EnumData.nextElement());
		
	} EnumData= reduced.enumerateInstances();
	while(EnumData.hasMoreElements())
		push((Instance) EnumData.nextElement());
  }
	   
  /**
   * Main method for running this filter.
   *
   * @param args 	should contain arguments to the filter: 
   * 			use -h for help
   */
 public static void main(String[] args) {
    runFilter(new WilsonEditing(), args);
  }
}
