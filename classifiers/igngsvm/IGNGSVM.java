package moa.classifiers.igngsvm;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Currency;

import moa.classifiers.AbstractClassifier;
import moa.clusterers.gng.GNG;
import moa.clusterers.gng.Gprototipo;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.WEKAClassOption;
import weka.classifiers.*;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;

public class IGNGSVM extends AbstractClassifier {
	  	
	private static final long serialVersionUID = 1L;

	/*
	 * Modifications by Andres Leon Suarez Cetrulo:
	 * - Implementation of Incremental GNG SVM in LibSVM classes.
	 */
	
	public IntOption lambdaOption = new IntOption("lambda", 'l', "Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm',"MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", 'a',"Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", 'd',"d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", 'e',"EpsilonB", 0.2);
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'n',"EpsilonN", 0.006);
	public IntOption nParadaOption = new IntOption("criterioParada", 'c', "Criterio de parada", 100);
	
	public IntOption tsOption = new IntOption("blocksSize", 't',"BlockSize",10100);
    public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'b',
            "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.functions.LibSVM");

	//Objeto SVM
	public LibSVM svm;

	//Prototype set
	private ArrayList<ArrayList<Gprototipo>> S;
	
	//Support Vector set
	public ArrayList<Gprototipo> SV;
		
	public double sumaDeMedias, mediaGlobal;
	
	public int iteracion = 1;
	
	//nle
    protected int numberInstances;
    
    //Bloque de datos TS
    protected Instances TS;
    
    //Lista de bloques de datos con distintas etiquetas
	public ArrayList<Instances> TSi;
            
    protected Instances instancesBuffer;

    protected boolean isClassificationEnabled;

    protected boolean isBufferStoring;
    
    //double tiempo;
    	
	public boolean isRandomizable() {
		return false;
	}	
	
	//preparing for learning
	public void resetLearningImpl() {
		SV = new ArrayList<Gprototipo>();
		S = new ArrayList<ArrayList<Gprototipo>>();
		
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
		
		//Instances GNG list
		ArrayList<GNG> GNGi = new ArrayList<GNG>();
		ArrayList<Double> labels = new ArrayList<Double>();
		
		
    	try {
            if (numberInstances == 0) {
            	//System.out.println("Comienza a entrenar el bloque");
           	 	//tiempo = System.currentTimeMillis();
           	 	//System.out.println("Comienza en tiempo 0");
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
                     //System.out.println(numberInstances);
                 }
                 
                 if(numberInstances % tsOption.getValue() == 0 && instancesBuffer.size() == tsOption.getValue()){
                	                    	 
			        /**Construimos TS con las instancias del buffer*/
                	 /*System.out.println();
                     System.out.println("El tamanyo de TS es: " + tsOption.getValue());
                     System.out.println();
                	 System.out.println("Nœmero de instancias entrenadas: "+numberInstances);
                	 System.out.println("Nœmero de instancias en este bloque: "+instancesBuffer.size());*/
 	                //System.out.println();
 	                //System.out.println("//////////////////////////////////////");
                	//System.out.println();
                	//System.out.println("Comienzo de bloque TS, de longitud de "+ tsOption.getValue() +" instancias");
			    	TS = new Instances(instancesBuffer);
			    	TSi = new ArrayList<Instances>();
			        isBufferStoring = false;
			        //isClassificationEnabled = true;
					S.clear();                	 
					

					/** Se divide el bloque TS en tantos bloques TSi como numero i de clases*/
					//Recorremos el bloque de instancias para separar estas en subconjuntos dependiendo de su clase
                	for (int i = 0; i < TS.numInstances(); i++) {
    					Instance aux = (Instance) TS.get(i).copy();
    					double auxLabel = aux.classValue();
    					
                		    					
                		//Si la clase ya existe se agrega a su subconjunto de datos
                		if(labels.contains(auxLabel)){
                			////System.out.println("ENTRA AQUI:");
        					Instances TsiAux = TSi.get(labels.indexOf(auxLabel));
                			TsiAux.add(aux);
                			TSi.set(labels.indexOf(auxLabel),TsiAux);
                			////System.out.println("SE METE EN: "+labels.indexOf(auxLabel)+ " de TsiAux");
                			
                    	//Si no existe se agrega a la lista de subconjuntos
                		}else{
        					Instances TsiAux = new Instances (((Instance)(inst.copy())).dataset());
        					TsiAux.clear();
                			TsiAux.add(aux);
                			TSi.add(TsiAux);
                			labels.add(auxLabel);
                		}
                		
                		//System.out.println("TRAZA NUMERO DE CLASES: "+TSi.size());                		
                		//System.out.println();
                		
					}
                	

                	/** Una vez separados los datos en sub-conjuntos se crean i objetos GNG distintos que se entrenaran 
                	 * con sus bloques de datos correspondientes dependiendo de la clase */          		
            		for (int i = 0; i < TSi.size(); i++) {
            			
            			//Se crea el clasificador GNGi
                   		GNG aux = new GNG();
                   		
                   		//Se reinicia el entrenamiento
                    	aux.resetLearningImpl();
                    	
                    	//Se envian los parametros de GNG
                    	aux.lambdaOption = this.lambdaOption;
                    	aux.alfaOption = this.alfaOption;
                    	aux.maxAgeOption = this.maxAgeOption;
                    	aux.constantOption = this.constantOption;
                    	aux.BepsilonOption = this.BepsilonOption;
                    	aux.NepsilonOption = this.NepsilonOption;
                    	aux.nParadaOption = this.nParadaOption;

                    	if(TSi.get(i).numInstances()==1){
                        	aux.trainOnInstanceImpl(TSi.get(i).get(0));

                    	}else if(TSi.get(i).numInstances()>1){
                        	//Anyadimos las instancias correspondientes segun su clase a cada objeto GNG y se entrena
                            for(int j=0;aux.getCreadas()<nParadaOption.getValue();j++){                        	
                            	aux.trainOnInstanceImpl(TSi.get(i).get(j));
                            	
                            	if(j+1==TSi.get(i).numInstances())
                            		j = -1;
                            }
                    	}
                    	
                    	/**Se agrega el GNGi correspondiente a la lista de GNGi 
                        y su conjunto S a la lista de subconjuntos de S*/
                        GNGi.add(aux);
                        //System.out.println(GNGi.get(i).getS());
                        S.add(GNGi.get(i).getS()); 
                        ////System.out.println("Agregado a S*");
                        
					}
            		
                    System.out.println();
	        		System.out.println();
	        		System.out.println("LISTA DE S: ");
	        		System.out.println();
	        		for (int i = 0; i < S.size(); i++) {
	        			for (int j = 0; j < S.get(i).size(); j++) {
			        		System.out.println(S.get(i).get(j)+", "+labels.get(i));

						}

					}
                    System.out.println();
					
                    /**Todos los i sub-bloques de TS han sido procesados*/
                    System.out.println("Inicio bloque TS - Se han procesado sus subconjuntos en instancias de GNG separadas");
            		System.out.println();
            		
	        		Temp = new Instances (inst.dataset());
	        		Temp.clear();

	        		SVpnew = null;
	        		

	        		/**Se combinan todos los conjuntos Si en S* */
	        		double media = 0;
					int contador = 0;	
	        		for (int i = 0; i < S.size(); i++) {
						
		        		System.out.println("Numero de instancias de la topologia GNG"+ i +": "+S.get(i).size());
	        			media += S.get(i).size();
		        		
		        		
		        		for (int k = 0; k < S.get(i).size(); k++, contador++) {
			        		Instance instS = (Instance) inst.copy();
	
			        		//Se obtienen los vectores de referencia W
		        			for (int j = 0; j < S.get(i).get(k).w.length; j++) {
				        		instS.setValue(j,S.get(i).get(k).w[j]);
				        		System.out.print(S.get(i).get(k).w[j]+ " ,");
							}
		        			
		        			//Se extienden los vectores de referencia W con la etiqueta del subgrupo GNGi
							instS.setClassValue(labels.get(i));
							Temp.add(instS);		        			
							System.out.println(labels.get(i));	        		
		        		}
	        		}media /= S.size();
	        		media = media/tsOption.getValue();
	        		sumaDeMedias += media;
	        		mediaGlobal = sumaDeMedias/iteracion;

	        		iteracion++;
	        		
	        		/** Se combina S* con los SV de la pasada iteracion*/
	        		System.out.println();
	        		for (int i = contador; i < SV.size()+contador; i++) {
	        			Instance instSV = (Instance) inst.copy();
	        			
	        			for (int j = 0; j < SV.get(i-contador).w.length; j++) {
		        			instSV.setValue(j,SV.get(i-contador).w[j]);

						}instSV.setClassValue(SV.get(i-contador).getClase());
						
						Temp.add(instSV);
						
	        		}//System.out.println();
	        		try {
	        			/**Train a new SVM with the set Temp*/
	        			buildClassifier(Temp);
	        			System.out.println("Crea de forma satisfactoria el clasificador GNGSVM");
	        			System.out.println("Longitud instancesBuffer "+instancesBuffer.numInstances());	        
                        isClassificationEnabled = true;
                        instancesBuffer = new Instances (((Instance)(inst.copy())).dataset());
                        instancesBuffer.clear();
                        isBufferStoring = true;

	        		} catch (Exception e) {
	        			e.printStackTrace();
	        		} 

	               	// System.out.println("4 El tiempo desde el comienzo del paso es: "+(System.currentTimeMillis()-tiempo));

	        		//System.out.println("Obtiene los datos de SV");
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
        				SV.add(new Gprototipo(SVpnew[i],claseSV[i]));
        				
	        		}
	        		
	        		
	        		System.out.println();
	        		System.out.println("LISTA DE SV: ");
	        		System.out.println();
	        		for (int i = 0; i < SV.size(); i++) {
		        		System.out.println(SV.get(i)+","+claseSV[i]);

					}
	        		System.out.println();
	        		System.out.println("La media de reduccion en esta iteracion es: "+(mediaGlobal));
	        		System.out.println();	        		
	        		System.out.println();
	                System.out.println("Fin bloque TS");
	                System.out.println();
	                System.out.println("//////////////////////////////////////");
	                System.out.println();
               	 //System.out.println("5 El tiempo desde el comienzo del paso es: "+(System.currentTimeMillis()-tiempo));

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
        	//System.out.println("Warning: ÁClasifica aleatoriamente! a la altura de "+numberInstances+" entrenadas");

            for (int i = 0; i < inst.numClasses(); i++) {
            	votes[i] = 1.0 / inst.numClasses();
            }
        } else {
            try {
            	//System.out.println("Clasifica bien a la altura de "+numberInstances+" entrenadas");
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
        f = o.getClass().getField(name);
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
         out.append(("Incremental GNGSVM by Andres Leon Suarez Cetrulo. "+svm.toString()));
    }
    
	protected moa.core.Measurement[] getModelMeasurementsImpl() {
		return null;
	}

}
