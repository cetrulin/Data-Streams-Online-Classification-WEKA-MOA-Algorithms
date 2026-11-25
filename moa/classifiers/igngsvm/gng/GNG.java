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

import java.util.ArrayList;
import weka.core.Instance;

import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.core.Measurement;
import moa.options.FloatOption;
import moa.options.IntOption;

/**
 * Growing Neural Gas (GNG) incremental topology-preserving clusterer for MOA.
 *
 * <p>Implements Bernd Fritzke's (1995) Growing Neural Gas algorithm with
 * undirected edge aging/pruning and optimized squared-distance nearest-unit
 * search.</p>
 *
 * <p>Reference:<br>
 * Bernd Fritzke. A Growing Neural Gas Network Learns Topologies.
 * Advances in Neural Information Processing Systems (NIPS), 1995.</p>
 *
 * <p>Parameters:</p>
 * <ul>
 * <li>-l : Lambda — number of steps between new unit insertions.</li>
 * <li>-m : Maximum edge age before pruning.</li>
 * <li>-a : Error reduction factor {@code alpha} on unit insertion.</li>
 * <li>-d : Global error decay factor per step.</li>
 * <li>-e : Winner learning rate {@code epsilon_b}.</li>
 * <li>-n : Neighbor learning rate {@code epsilon_n}.</li>
 * <li>-c : Maximum number of prototypes to create (stopping criterion).</li>
 * </ul>
 *
 * @author Andres L. Suarez-Cetrulo
 */
public class GNG extends AbstractClusterer {

	private static final long serialVersionUID = -8566293434212159290L;

	private ArrayList<GUnit> S;
	public IntOption lambdaOption = new IntOption("lambda", 'l', "Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm', "MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", 'a', "Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", 'd', "d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", 'e', "EpsilonB", 0.2);
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'n', "EpsilonN", 0.006);
	public IntOption stoppingCriteriaOption = new IntOption("stoppingCriteria", 'c', "Stopping criteria", 100);

	private long instancesSeen;
	private int neuronsCreated;
	private double lastNearestDist;

	/**
	 * Finds the two closest prototype units (winner {@code s1} and runner-up {@code s2})
	 * to {@code pattern} using squared Euclidean distance, caching the Euclidean
	 * distance to {@code s1} in {@code lastNearestDist}.
	 */
	public GUnit[] getNearest(double[] pattern) {
		GUnit first, second;
		GUnit[] p = new GUnit[2];

		if (this.S.size() >= 2) {
			double d0 = GUnit.distSquared(this.S.get(0).w, pattern);
			double d1 = GUnit.distSquared(this.S.get(1).w, pattern);
			double firstDistSq, secondDistSq;
			if (d0 <= d1) {
				first = this.S.get(0);
				firstDistSq = d0;
				second = this.S.get(1);
				secondDistSq = d1;
			} else {
				first = this.S.get(1);
				firstDistSq = d1;
				second = this.S.get(0);
				secondDistSq = d0;
			}
			final int size = this.S.size();
			for (int j = 2; j < size; j++) {
				GUnit candidate = this.S.get(j);
				double dSq = GUnit.distSquared(candidate.w, pattern);
				if (dSq < firstDistSq) {
					second = first;
					secondDistSq = firstDistSq;
					first = candidate;
					firstDistSq = dSq;
				} else if (dSq < secondDistSq) {
					second = candidate;
					secondDistSq = dSq;
				}
			}
			this.lastNearestDist = Math.sqrt(firstDistSq);
			p[0] = first;
			p[1] = second;
		} else {
			if (this.S.size() > 0) {
				p[0] = this.S.get(0);
				p[1] = this.S.get(0);
				this.lastNearestDist = GUnit.dist(p[0].w, pattern);
			} else {
				return null;
			}
		}
		return p;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	/**
	 * Returns the current list of prototype units in the GNG topology.
	 */
	public ArrayList<GUnit> getS() {
		return this.S;
	}

	@Override
	public Clustering getClusteringResult() {
		return null;
	}

	@Override
	public void resetLearningImpl() {
		this.S = new ArrayList<GUnit>();
		this.instancesSeen = 0;
		this.neuronsCreated = 0;
	}

	/**
	 * Returns the total number of interpolated prototype units created so far.
	 */
	public int getNumberOfPrototypesCreated() {
		return this.neuronsCreated;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		this.instancesSeen++;

		// 1. Input distribution (excluding class attribute)
		double[] pattern = new double[inst.numValues() - 1];
		for (int i = 0; i < pattern.length; i++) {
			pattern[i] = inst.value(i);
		}

		if (GUnit.getNumberOfGPrototypes(this.S) < 2) {
			// Stored with label 0 since GNG is unsupervised (each class is modeled in its own GNG instance)
			GUnit p = new GUnit(pattern, 0);
			if (!this.S.contains(p)) {
				this.S.add(p);
			}
		} else if (this.neuronsCreated < this.stoppingCriteriaOption.getValue()) {

			// 2. Find the two nearest units s1 and s2
			GUnit[] best = getNearest(pattern);
			GUnit s1 = best[0];
			GUnit s2 = best[1];

			// 3. Increment the age of all edges emanating from s1
			s1.updateAges();

			// 4. Accumulate local error for s1
			s1.setError(s1.getError() + this.lastNearestDist);

			// 5. Move s1 and its direct topological neighbors toward pattern using epsilon_b and epsilon_n
			s1.updateW(this.BepsilonOption.getValue(), pattern);
			final double nepsilon = this.NepsilonOption.getValue();
			final int nSize = s1.neighbors.size();
			for (int j = 0; j < nSize; j++) {
				GEdge edge = s1.neighbors.get(j);
				GUnit nb = (edge.p0 != s1) ? edge.p0 : edge.p1;
				nb.updateW(nepsilon, pattern);
			}

			// 6. Connect s1 and s2 (or reset existing edge age to 0)
			GEdge c = s1.searchEdge(s2);
			if (c == null) {
				s1.addNeighbor(s2);
			} else {
				c.age = 0;
			}

			// 7. Remove edges exceeding maxAge and purge any resulting isolated units
			if (s1.purgeGEdgesAndCheck(this.maxAgeOption.getValue())) {
				for (int k = 0; k < this.S.size(); k++) {
					if (this.S.get(k).neighbors.size() == 0) {
						this.S.remove(k);
						--k;
					}
				}
			}

			// 8. Every lambda steps, insert a new unit r between the highest-error unit q and its worst neighbor f
			if (this.instancesSeen % this.lambdaOption.getValue() == 0 && this.instancesSeen > 0) {
				// Find unit q with the maximum accumulated error
				double greatest = 0;
				int posGreatest = 0;
				for (int i = 0; i < this.S.size(); i++) {
					GUnit current = this.S.get(i);
					double error = current.getError();
					if (error > greatest) {
						greatest = error;
						posGreatest = i;
					}
				}
				GUnit q = this.S.get(posGreatest);

				// Find neighbor f of q with the maximum error
				GUnit f;
				double error;
				greatest = 0;
				int posGreatestNeighbor = 0;
				for (int i = 0; i < q.neighbors.size(); i++) {
					GEdge current = q.neighbors.get(i);
					if (current.p0.equals(q)) {
						error = current.p1.getError();
					} else {
						error = current.p0.getError();
					}
					if (error > greatest) {
						greatest = error;
						posGreatestNeighbor = i;
					}
				}
				if (q.neighbors.get(posGreatestNeighbor).p0.equals(q)) {
					f = q.neighbors.get(posGreatestNeighbor).p1;
				} else {
					f = q.neighbors.get(posGreatestNeighbor).p0;
				}

				// Interpolate new prototype r halfway between q and f
				double[] wr = new double[q.w.length];
				for (int i = 0; i < wr.length; i++) {
					wr[i] = 0.5 * (q.w[i] + f.w[i]);
				}
				GUnit r = new GUnit(wr, 0);

				// Connect r to q and f, and remove edge q-f
				r.addNeighbor(q);
				r.addNeighbor(f);
				q.removeNeighbor(f);

				// Decrease error variables of q and f by factor alpha
				q.setError(q.getError() * this.alfaOption.getValue());
				f.setError(f.getError() * this.alfaOption.getValue());

				// Initialize error of r with the updated error of q
				r.setError(q.getError());

				this.S.add(r);
				this.neuronsCreated++;
			}

			// 9. Decay all unit error variables by factor d
			for (int i = 0; i < this.S.size(); i++) {
				GUnit current = this.S.get(i);
				current.setError(current.getError() * this.constantOption.getValue());
			}
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("Growing Neural Gas for MOA, implemented by Andres L. Suarez-Cetrulo.");
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return null;
	}
}
