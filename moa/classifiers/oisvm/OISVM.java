package moa.classifiers.oisvm;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.ilvq.Conexion;
import moa.classifiers.ilvq.Ilvq;
import moa.classifiers.ilvq.Prototipo;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.WEKAClassOption;
import weka.classifiers.*;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;

public class OISVM extends AbstractClassifier {
	  	
	private static final long serialVersionUID = 1L;

	/*
	 * Modifications by Andres Leon Suarez Cetrulo:
	 * - Implementation of Online Incremental SVM in LibSVM classes.
	 */
	
	 /**<!-- technical-bibtex-start -->
	 * BibTeX:
	 * <pre>
	 *&#64;misc{title = {An Online Incremental Learning Support Vector Machine for Large-Scale Data*},
	 * author = {Jun Zheng, Hui Yu, Furao Shen, and Jinxi Zhao}
	 * note ={ - National Key Laboratory for Novel Software Technology, Nanjing University, China 
	  - Jiangyin Information Technology Research Institute, Nanjing University, China 
	  - junzheng@smail.nju.edu.cn, {frshen,jxzhao}@nju.edu.cn}
	 *
	 * &#64;misc{EL-Manzalawy2005,
	 *    author = {Yasser EL-Manzalawy},
	 *    note = {You don't need to include the WLSVM package in the CLASSPATH},
	 *    title = {WLSVM},
	 *    year = {2005},
	 *    URL = {http://www.cs.iastate.edu/\~yasser/wlsvm/}
	 * }
	 * 
	 * &#64;misc{Chang2001,
	 *    author = {Chih-Chung Chang and Chih-Jen Lin},
	 *    note = {The Weka classifier works with version 2.82 of LIBSVM},
	 *    title = {LIBSVM - A Library for Support Vector Machines},
	 *    year = {2001},
	 *    URL = {http://www.csie.ntu.edu.tw/\~cjlin/libsvm/}
	 * }
	 * </pre>
	 * <p/>
	 <!-- technical-bibtex-end -->/*
	 */


	public FloatOption lambdaOption = new FloatOption("lambda", 'l', "Lambda", 16);
	public IntOption maxAgeOption = new IntOption("maxAge", 'a',"MaximumAge", 16);
	public IntOption tsOption = new IntOption("blocksSize", 't',"BlockSize",300);
	
    public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'b',
            "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.functions.LibSVM");

	//Objeto SVM
	public LibSVM svm;

	//Prototype set
	private ArrayList<Prototipo> G;
	
	//Support Vector set
	public ArrayList<Prototipo> SV;	
	
	public double sumaDeMedias, mediaGlobal;
	
	public int iteracion = 1;

	//nle
    protected int numberInstances;
    
    //Bloque de datos TS
    protected Instances TS;
            
    protected Instances instancesBuffer;

    protected boolean isClassificationEnabled;

    protected boolean isBufferStoring;
    	
	public boolean isRandomizable() {
		return false;
	}	
	
	//preparing for learning
	public void resetLearningImpl() {
		SV = new ArrayList<Prototipo>();
		G = new ArrayList<Prototipo>();
		
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
        
    	//Variables auxiliares para el metodo
		double [][] SVpnew = null;
		int []label, nSV, claseSV; 
		int anterior;
		
		//Instancias de otras clases auxiliares
		Instances Temp;
		Ilvq obj;
		    	
    	try {
            if (numberInstances == 0) {
                this.instancesBuffer = new Instances (((Instance)(inst.copy())).dataset());
                this.instancesBuffer.clear();
                
                if (svm instanceof UpdateableClassifier) {
                    svm.buildClassifier(instancesBuffer);
                    this.isClassificationEnabled = true;
                } else {
                    this.isBufferStoring = true;
                }
                
            }numberInstances++;

            if (svm instanceof UpdateableClassifier) {
                if (numberInstances > 0) {
                    ((UpdateableClassifier) svm).updateClassifier(inst);
                }
            } else {

                 if (isBufferStoring == true) {
                     instancesBuffer.add(inst);
                 }
                 
                 if(numberInstances % tsOption.getValue() == 0 && instancesBuffer.size() == tsOption.getValue()){              	 
                	 System.out.println();
                     System.out.println("El tamanyo de TS es: " + tsOption.getValue());
                     System.out.println();
                	 System.out.println("Nœmero de instancias entrenadas: "+numberInstances);
                	 System.out.println("Nœmero de instancias en este bloque: "+instancesBuffer.size());
                               	 
                    /**Construimos TS con las instancias del buffer*/
                	TS = new Instances(instancesBuffer);
                    isBufferStoring = false;
                    isClassificationEnabled = true;               	
            		G.clear();
            		
                	obj = new Ilvq();
                	obj.resetLearningImpl();
                	obj.lambdaOption = this.lambdaOption;
                	obj.maxAgeOption = this.maxAgeOption;

                    for(int i=0;i<TS.numInstances();i++){                   	
                    	/**input a new pattern to the system*/
                    	obj.trainOnInstanceImpl(TS.get(i));

                    } G = obj.getG();  
                    
                    System.out.println("El tamanyo de G es: " + G.size());
	        		double media = G.size();
	        		
	        		
	        		media = media/tsOption.getValue();
	        		
	        		sumaDeMedias += media;
	        		
	        		mediaGlobal = sumaDeMedias/iteracion;
	        		
	        		iteracion++;
                    
                    /**All the data in the block TS has been processed*/
                    //System.out.println("Inicio bloque TS");

	        		Temp = new Instances (((Instance)(inst.copy())).dataset());
	        		Temp.clear();
	        		SVpnew = null;
	        				        		
	        		/**Combine the prototype set G and the support vector set SV in new set Temp*/
	        		for (int i = 0; i < G.size(); i++) {
		        		Instance instG = (Instance) inst.copy();

	        			for (int j = 0; j < G.get(i).w.length; j++) {
			        		instG.setValue(j,G.get(i).w[j]);
						}

						instG.setClassValue(G.get(i).getClase());
						Temp.add(instG);		        			
	        		
	        		}////System.out.println();
	        		for (int i = G.size(); i < SV.size()+G.size(); i++) {

	        			Instance instSV = (Instance) inst.copy();
	        			
	        			for (int j = 0; j < SV.get(i-G.size()).w.length; j++) {
		        			instSV.setValue(j,SV.get(i-G.size()).w[j]);

						}

						instSV.setClassValue(SV.get(i-G.size()).getClase());
						Temp.add(instSV);
	        		}//System.out.println();
	
	        		try {
	        			/**Train a new SVM with the set Temp*/
	        			buildClassifier(Temp);
	        				        			
                        isClassificationEnabled = true;
                        instancesBuffer = new Instances (((Instance)(inst.copy())).dataset());
                        instancesBuffer.clear();
                        isBufferStoring = true;

	        		} catch (Exception e) {
	        			e.printStackTrace();
	        		} 

	        		/**Get the new support vector set SVpnew*/	        	    				        			
        			label =  (int[]) getField (svm.m_Model,"label");
        			nSV =  ((int[]) getField (svm.m_Model,"nSV"));        			
        			Object [][]o =  (Object[][]) getField (svm.m_Model,"SV");
        			SVpnew = new double[o.length][];       			
        			claseSV = new int [SVpnew.length]; 
        			
        			for (int i = 0; i < o.length; i++) {
        				SVpnew [i]= new double [o[i].length]; 
						for (int j = 0; j < o[i].length; j++) {
							SVpnew[i][j] =((Double) getField (o[i][j],"value"));
							
						}anterior = 0;
						for(int j = 0;j<nSV.length;){
							if (i<nSV[j]+anterior){
								claseSV[i]=label[j];
								j = nSV.length;

							}else {
								anterior+=nSV[j];
								j++;
							}
						}
					}
        			
        			/**Update the old support vector set SV*/
	        		SV.clear();
	        		for(int i = 0;i<SVpnew.length;i++){

        				SV.add(new Prototipo(SVpnew[i],claseSV[i]));
        				
	        		}
	        		
	        		System.out.println("La media de reduccion en esta iteracion es: "+(mediaGlobal));
	                //System.out.println("Fin bloque TS");
	                //System.out.println("//////////////////////////////////////");

                }	
            } 

        } catch (Exception e) {
            System.err.println("Training: " + e.getMessage());
            e.printStackTrace();
        }
        
    } 
    
	//predict class	
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
     * returns the current value of the specified field.
     * 
     * @param o           the object the field is member of
     * @param name        the name of the field
     * @return            the value
     */
    protected Object getField(Object o, String name) {
      Field       f;
      Object      result;
      
      try {
        f      = o.getClass().getField(name);
        result = f.get(o);
      }
      catch (Exception e) {
        e.printStackTrace();
        result = null;
      }
      
      return result;
    }

    //////////////////////////////////
	
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
         out.append(("Online Incremental SVM by Andres Leon Suarez Cetrulo. "+svm.toString()));
    }
    
	protected moa.core.Measurement[] getModelMeasurementsImpl() {
		return null;
	}

}
