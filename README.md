OISVM*
===
OI-SVM implementation for MOA, proposed in http://dx.doi.org/10.1007/978-3-642-15822-3_9

IGNGSVM*
===
Implementation of iGNGSVM used in https://doi.org/10.1016/j.neucom.2016.12.093

**Both OISVM and IGNGSVM use the LibSVM for WEKA jar file present in the root of this repo. Support Vectors are retrieved from the SVM using JAR's classes objects.

GNG
===
Growing Neural Gas is implemented in MOA as topology generation method for IGNGSVM. Proposed by Fritzke in 1995 https://papers.nips.cc/paper/893-a-growing-neural-gas-network-learns-topologies.pdf 

Wilson Edited (Also known as Edited Nearest-Neighbor)
===
Noise reduction algorithm based in Nearest Neighbors. Proposed in https://doi.org/10.1109/TSMC.1972.4309137

Task & Evaluation
====
It evaluates test and train in parallel to allow online evaluation for non-stationary environments in presence of concept drift.


