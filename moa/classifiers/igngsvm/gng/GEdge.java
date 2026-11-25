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

/**
 * Undirected aged edge connecting two {@link GUnit} prototypes in a
 * Growing Neural Gas (GNG) topology.
 */
public class GEdge implements Serializable {

	private static final long serialVersionUID = 1L;

	public GUnit p0;
	public GUnit p1;
	public int age;

	public GEdge(GUnit p0, GUnit p1) {
		this.p0 = p0;
		this.p1 = p1;
		this.age = 0;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((this.p0 == null) ? 0 : this.p0.hashCode());
		result = prime * result + ((this.p1 == null) ? 0 : this.p1.hashCode());
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
		GEdge other = (GEdge) obj;
		if (this.p0 == null) {
			if (other.p0 != null)
				return false;
		} else if (!this.p0.equals(other.p0))
			return false;
		if (this.p1 == null) {
			if (other.p1 != null)
				return false;
		} else if (!this.p1.equals(other.p1))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "GEdge [p0=" + this.p0 + ", p1=" + this.p1 + "] :" + this.age;
	}
}
