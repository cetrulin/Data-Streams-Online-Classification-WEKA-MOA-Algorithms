package moa.clusterers.gng;

/**
	Version modificada para conexion.java incluido en ILVQ.jar 
	por Andrés León Suárez Cetrulo
*/

public class Gconexion {
	public Gprototipo p0;
	public Gprototipo p1;
	public int edad;
	
	public Gconexion(Gprototipo p0, Gprototipo p1){
		this.p0 = p0;
		this.p1 = p1;
		edad = 0;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((p0 == null) ? 0 : p0.hashCode());
		result = prime * result + ((p1 == null) ? 0 : p1.hashCode());
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
		Gconexion other = (Gconexion) obj;
		if (p0 == null) {
			if (other.p0 != null)
				return false;
		} else if (!p0.equals(other.p0))
			return false;
		if (p1 == null) {
			if (other.p1 != null)
				return false;
		} else if (!p1.equals(other.p1))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "Gconexion [p0=" + p0 + ", p1=" + p1 + "] :" +edad;
	}
	
}
