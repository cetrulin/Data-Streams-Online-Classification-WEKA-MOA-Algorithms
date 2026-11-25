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

package moa.classifiers.igngsvm.gng;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Prototype node (unit) in a Growing Neural Gas (GNG) topology graph.
 */
public class GUnit implements Serializable {

	private static final long serialVersionUID = 1L;

	public double[] w;
	private int label;
	private double errorAccum;

	public ArrayList<GEdge> neighbors = new ArrayList<GEdge>();

	public GUnit(double[] w, int label) {
		this.w = w.clone();
		this.label = label;
		this.errorAccum = 0;
	}

	public void setError(double error) {
		this.errorAccum = error;
	}

	public double getError() {
		return this.errorAccum;
	}

	public int getLabel() {
		return this.label;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + this.label;
		result = prime * result + Arrays.hashCode(this.w);
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
		GUnit other = (GUnit) obj;
		if (this.label != other.label)
			return false;
		if (!Arrays.equals(this.w, other.w))
			return false;
		return true;
	}

	/**
	 * Moves this unit's weight vector toward {@code pattern} with learning rate {@code ep}.
	 */
	public void updateW(double ep, double[] pattern) {
		for (int i = 0; i < this.w.length; i++) {
			this.w[i] = this.w[i] + ep * (pattern[i] - this.w[i]);
		}
	}

	/**
	 * Returns the number of prototypes in {@code list} in O(1) time.
	 */
	public static int getNumberOfGPrototypes(ArrayList<GUnit> list) {
		return list.size();
	}

	/**
	 * Computes the squared Euclidean distance between two weight vectors.
	 */
	public static double distSquared(double[] w1, double[] w2) {
		double sum = 0;
		for (int i = 0; i < w1.length; i++) {
			sum += Math.pow(w1[i] - w2[i], 2);
		}
		return sum;
	}

	/**
	 * Computes the Euclidean distance between two weight vectors.
	 */
	public static double dist(double[] w1, double[] w2) {
		return Math.sqrt(distSquared(w1, w2));
	}

	/**
	 * Adds an undirected edge between this unit and {@code s2}.
	 */
	public void addNeighbor(GUnit s2) {
		GEdge edge = new GEdge(this, s2);
		this.neighbors.add(edge);
		s2.neighbors.add(edge);
	}

	@Override
	public String toString() {
		return Arrays.toString(this.w) + ":" + this.label;
	}

	/**
	 * Searches for an existing edge connecting this unit to {@code s2}.
	 */
	public GEdge searchEdge(GUnit s2) {
		for (GEdge c : this.neighbors) {
			if (c.p1.equals(s2) || c.p0.equals(s2)) {
				return c;
			}
		}
		return null;
	}

	/**
	 * Increments the age of all edges incident to this unit by 1.
	 */
	public void updateAges() {
		for (GEdge c : this.neighbors) {
			c.age++;
		}
	}

	/**
	 * Removes all incident edges whose age is greater than or equal to {@code maxAge}.
	 */
	public void purgeGEdges(double maxAge) {
		purgeGEdgesAndCheck(maxAge);
	}

	/**
	 * Removes all incident edges whose age is greater than or equal to {@code maxAge}
	 * from both endpoints and returns {@code true} if at least one edge was removed.
	 */
	public boolean purgeGEdgesAndCheck(double maxAge) {
		boolean removedAny = false;
		for (int i = 0; i < this.neighbors.size(); i++) {
			GEdge c = this.neighbors.get(i);
			if (c.age >= maxAge) {
				removedAny = true;
				this.neighbors.remove(i);
				i--;
				if (c.p0.equals(this)) {
					c.p1.neighbors.remove(c);
				} else {
					c.p0.neighbors.remove(c);
				}
			}
		}
		return removedAny;
	}

	/**
	 * Removes the undirected edge between this unit and {@code p} from both endpoints.
	 */
	public void removeNeighbor(GUnit p) {
		for (int i = 0; i < this.neighbors.size(); i++) {
			GEdge c = this.neighbors.get(i);
			if (c.p0.equals(p) || c.p1.equals(p)) {
				this.neighbors.remove(i);
				i--;
			}
		}
		for (int i = 0; i < p.neighbors.size(); i++) {
			GEdge c = p.neighbors.get(i);
			if (c.p0.equals(this) || c.p1.equals(this)) {
				p.neighbors.remove(i);
				i--;
			}
		}
	}

	/**
	 * Returns an array of all neighboring units connected to this unit.
	 */
	public GUnit[] getNeighborhood() {
		GUnit[] v = new GUnit[this.neighbors.size()];
		for (int i = 0; i < v.length; i++) {
			if (this.neighbors.get(i).p0 != this) {
				v[i] = this.neighbors.get(i).p0;
			} else {
				v[i] = this.neighbors.get(i).p1;
			}
		}
		return v;
	}
}
