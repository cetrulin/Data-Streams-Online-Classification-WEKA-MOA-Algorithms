package moa.clusterers.gng;

import java.util.ArrayList;
import java.util.Arrays;
/**
	Version modificada para de prototipo.java incluido en ILVQ.jar 
	por Andrés León Suárez Cetrulo
*/

public class Gprototipo {
	public double w[];
	private int clase;	
	private double errorAcumulado;
	
	public ArrayList<Gconexion> vecinos = new ArrayList<Gconexion>();
	
	public Gprototipo(double w[], int clase){
			this.w = w.clone();
			this.clase = clase;
			this.errorAcumulado = 0;
	}
	
	public void setError(double error){
			this.errorAcumulado = error;		
	}
	
	public double getError(){
		return errorAcumulado;
	}
	
	public int getClase(){
		return clase;
	}
	/*public Instance getInstance(){
		return instance;
	}*/

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + clase;
		result = prime * result + Arrays.hashCode(w);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Gprototipo other = (Gprototipo) obj;
		if (clase != other.clase)
			return false;
		if (!Arrays.equals(w, other.w))
			return false;
		return true;
	}
	
	public void actualizarW(double ep,double patron[]){
		for (int i = 0; i < w.length; i++) {
			w[i] = w[i] + ep*(patron[i] - w[i]);
		}
	}
		
	public static int numeroGprototipos(ArrayList<Gprototipo> lista){
		int n=0;
		for (Gprototipo p : lista) {
			n++;
		}
		return n;
	}
	
	public static double dist(double w1[],double w2[]){
		double sum = 0;
		for (int i = 0; i < w1.length; i++) {
			sum += Math.pow(w1[i]-w2[i],2);
		}
		return Math.sqrt(sum);
	}
	
	public void anadirVecino(Gprototipo s2){
		Gconexion nueva = new Gconexion(this,s2);
		vecinos.add(nueva);
		s2.vecinos.add(nueva);
	}
	
	public String toString(){
		return Arrays.toString(w)+":"+clase;
	}
	
	public Gconexion buscarEnlace(Gprototipo s2){
		for (Gconexion c : vecinos) {
			if(c.p1.equals(s2)||c.p0.equals(s2)) return c;
		}
		return null;
	}
	
	public void actualizarEdades(){
		for (Gconexion c : vecinos) {
			c.edad++;
		}
	}
	
	public void purgarGconexiones(double edad){
		for (int i = 0;i<vecinos.size();i++){
			Gconexion c = vecinos.get(i);
			if(c.edad>=edad){
				vecinos.remove(c);
				if(c.p0.equals(this)){
					c.p1.vecinos.remove(c);
				}else{
					c.p0.vecinos.remove(c);
				}
			}
		}
	}
	
	public void eliminarVecino(Gprototipo p){
		for (int i = 0 ; i<vecinos.size();i++) {
			if(vecinos.get(i).p0.equals(p)||vecinos.get(i).p1.equals(p)) 
				vecinos.remove(vecinos.get(i));
		}
			
	}
	
	public Gprototipo[] recuperarVecindario(){
		Gprototipo v[] = new Gprototipo[vecinos.size()];
		for (int i = 0; i < v.length; i++) {
			if(vecinos.get(i).p0!=this)	
				v[i] = vecinos.get(i).p0;
			else 
				v[i] = vecinos.get(i).p1;
			
		}return v;
	}

}
