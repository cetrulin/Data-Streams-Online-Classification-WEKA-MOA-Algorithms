package moa.clusterers.gng;

import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.beans.Clusterer;

import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.core.Measurement;
import moa.options.FloatOption;
import moa.options.IntOption;

public class GNG extends AbstractClusterer {

	/**
	 * Algoritmo implementado, siguiendo la nomenclatura utilizada en ILVQ.jar
	 * por Andrés León Suárez Cetrulo
	 */
	private static final long serialVersionUID = -8566293434212159290L;

	private ArrayList<Gprototipo> S;	
	public IntOption lambdaOption = new IntOption("lambda", 'l', "Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm',"MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", 'a',"Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", 'd',"d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", 'e',"EpsilonB", 0.2);
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'n',"EpsilonN", 0.006);
	public IntOption nParadaOption = new IntOption("criterioParada", 'c', "Criterio de parada", 100);

	private long patrones_presentados;
	private int neuronasCreadas;	
	
	private Instance copia;
	
	public Gprototipo[] obtenerCercanos(double patron[]){
		Gprototipo ganador,segundon;
		Gprototipo p[]=new Gprototipo[2];
		if(S.size()>=2){
			if(Gprototipo.dist(S.get(0).w,patron)<=Gprototipo.dist(S.get(1).w,patron)){
				ganador = S.get(0);
				segundon = S.get(1);
			}else{
				ganador = S.get(1);
				segundon = S.get(0);
			}
			for (int j = 2; j < S.size(); j++) {
				if(Gprototipo.dist(S.get(j).w,patron)<Gprototipo.dist(ganador.w,patron)){
					segundon = ganador;
					ganador = S.get(j);
				}else{
					if(Gprototipo.dist(S.get(j).w,patron)<Gprototipo.dist(segundon.w,patron)){
						segundon = S.get(j);
					}
				}
				
			}p[0]=ganador;
			p[1]=segundon;
			
		}else{
			if(S.size()>0){
				p[0]=S.get(0);
				p[1]=S.get(0);
			}else{
				return null;
			}
		}
		return p;
	}
	

	@Override
	public boolean isRandomizable() {
		return false;
	}

	public ArrayList<Gprototipo> getS (){
		
		return S;	
	}

	public Clustering getClusteringResult() {

		return null;
	}
	
	public void log (String st)
  {
     try {
		PrintWriter pw = new PrintWriter(new FileWriter ("gng.log",true));		
		pw.println(st);
		pw.flush();
		pw.close ();
     }
     catch (Exception e)
    {  
    }
  }

	@Override
	public void resetLearningImpl() {
		S = new ArrayList<Gprototipo>();
		patrones_presentados = 0;
		neuronasCreadas = 0;
	}

	public int getCreadas() {		
		return neuronasCreadas;
	}
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		copia = (Instance) inst.copy();
		patrones_presentados++;
		//1. Distribucion de entrada
		double []patron =  new double[inst.numValues()-1];
		for (int i = 0; i < patron.length; i++) {
			patron[i]=inst.value(i);
		}
		/**********************************seguimos inicializando patrones*************************************************/
		
		if(Gprototipo.numeroGprototipos(S)<2){			
			//Se guarda como clase 0, ya que GNG no gestiona las clases. No es un algoritmo de clasificacion.
			Gprototipo p = new Gprototipo(patron,0);

			if(!S.contains(p))
				S.add(p);
			
		}else if(neuronasCreadas < nParadaOption.getValue()){
						
			//2. Se encuentra s1 y s2		
			Gprototipo []mejores;
			Gprototipo s1=null,s2=null;
			mejores = obtenerCercanos(patron);
			s1 = mejores[0];
			s2 = mejores[1];			
			
			//3. Se incrementa la edad de todas las conexiones de s1
			s1.actualizarEdades();
			
			//4. Variable local de acumulación de error de s1
			s1.setError(s1.getError()+s1.dist(s1.w.clone(),patron));			

			/*************************************actualizamos las posiciones de los prototipos**************************************/
	
			//5. Movemos s1 y sus vecinos con Eb y En
			
			s1.actualizarW(BepsilonOption.getValue(), patron);
			Gprototipo vecinos[] = s1.recuperarVecindario();
			for (int j = 0; j < vecinos.length; j++) {
				vecinos[j].actualizarW(NepsilonOption.getValue(), patron);
			}						
			
			/************************anadimos los nuevos enlaces o actualizamos los ya presentes***********************/
			
			//6. Reseteo de conexiones entre s1 y s2
			Gconexion c = s1.buscarEnlace(s2);
			if(c==null){
				s1.anadirVecino(s2);
			}else{;
				c.edad=0;
			}
			
			/******************************************comenzamos a eliminar prototipos y conexiones *********************************/
			
			//7.eliminamos una conexion cuando su edad supera el maximo permitido
			s1.purgarGconexiones(maxAgeOption.getValue());
			//Se eliminan los individuos sin conexiones
			for (int k = 0; k < S.size(); k++) {
				if(S.get(k).vecinos.size()==0){
					S.remove(k);
					--k;
				}
			}
			
			//8.si llegamos a una iteracion multiplo de lambda se interpola
			if(patrones_presentados%lambdaOption.getValue()==0&&patrones_presentados>0){
				
				//Se determina el punto q con mayor error
				double mayor = 0;
				int posMayor = 0;
				for(int i = 0; i < S.size(); i++){
					Gprototipo actual = S.get(i);
					double error = actual.getError();
					if(error>mayor){
						mayor = error;
						posMayor = i;
					}
				}Gprototipo q = S.get(posMayor);
				
				//Se calcula el vecino de q con mas error
				Gprototipo f;
				double error;
				mayor = 0;
				int posMayorVecino = 0;
				for (int i = 0; i < q.vecinos.size(); i++) {
					Gconexion actual = q.vecinos.get(i);
					error = 0;
					if(actual.p0.equals(q)){
						error = actual.p1.getError();
					}else{
						error = actual.p0.getError();
					}
					if(error>mayor){
						mayor = error;
						posMayorVecino = i;
					}
				}if(q.vecinos.get(posMayorVecino).p0.equals(q)){
					f = q.vecinos.get(posMayorVecino).p1;
				}else{
					f = q.vecinos.get(posMayorVecino).p0;
				}
				
				//Se coloca r entre q y el vecino f que mas error tenga
				double wr[] = new double [q.w.length];				
				for (int i = 0; i < wr.length; i++) {
					wr[i]=0.5*(q.w[i]+f.w[i]);
					
				}//se crea el nuevo prototipo (la clase es indiferente pues no es gestionada por GNG)
				Gprototipo r = new Gprototipo(wr,0);
				
				//Se conecta r con q y f y se elimina la conexion q-f
				r.anadirVecino(q);
				r.anadirVecino(f);
				q.eliminarVecino(f);
				
				//Error de q y f se descrementa multiplicandolo por alfa.
				double aux = q.getError();
				aux*=alfaOption.getValue();
				q.setError(aux);
				aux = f.getError();
				aux*=alfaOption.getValue();
				f.setError(aux);
				
				//Error r = nuevo error q
				r.setError(q.getError());
				
				//Se actualiza S
				S.add(r);
				neuronasCreadas++;
				
				//S.set(posMayor, q);
				//S.set(posMayorVecino, f);
				
			}
			
			//9.Se reducen las variables de error multiplicandolas por d.			
			for(int i = 0; i < S.size(); i++){
				Gprototipo actual = S.get(i);
				double error = actual.getError();
				error *= constantOption.getValue();
				actual.setError(error);
				//S.set(i, actual);
			}
			
		}	
		
//		System.out.println(G);
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
        out.append("Growing Neural Gas for MOA, Implement by Andres Leon Suarez Cetrulo.");		
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return null;
	}

}
