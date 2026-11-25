package moa.classifiers.igngsvm.gng;

import java.util.ArrayList;
import java.util.Arrays;

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
		return lista.size();
	}
	
	public static double distSquared(double w1[],double w2[]){
		double sum = 0;
		for (int i = 0; i < w1.length; i++) {
			sum += Math.pow(w1[i]-w2[i],2);
		}
		return sum;
	}

	public static double dist(double w1[],double w2[]){
		return Math.sqrt(distSquared(w1, w2));
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
		purgarGconexionesYComprobar(edad);
	}

	public boolean purgarGconexionesYComprobar(double edad){
		boolean removedAny = false;
		for (int i = 0;i<vecinos.size();i++){
			Gconexion c = vecinos.get(i);
			if(c.edad>=edad){
				removedAny = true;
				vecinos.remove(i);
				i--;
				if(c.p0.equals(this)){
					c.p1.vecinos.remove(c);
				}else{
					c.p0.vecinos.remove(c);
				}
			}
		}
		return removedAny;
	}
	
	public void eliminarVecino(Gprototipo p){
		for (int i = 0 ; i<this.vecinos.size();i++) {
			Gconexion c = this.vecinos.get(i);
			if(c.p0.equals(p)||c.p1.equals(p)) {
				this.vecinos.remove(i);
				i--;
			}
		}
		for (int i = 0 ; i<p.vecinos.size();i++) {
			Gconexion c = p.vecinos.get(i);
			if(c.p0.equals(this)||c.p1.equals(this)) {
				p.vecinos.remove(i);
				i--;
			}
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
