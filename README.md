
IGNGSVM
===
Implementation of [incremental Growing Neural Gas Support Vector Machines (iGNGSVM)](https://doi.org/10.1016/j.neucom.2016.12.093) for [MOA](http://moa.cms.waikato.ac.nz). See it in '/moa/classifiers'.

OISVM
===
Implementation of [Online Incremental Learning Support Vector Machine (OI-SVM)](http://dx.doi.org/10.1007/978-3-642-15822-3_9) for [MOA](http://moa.cms.waikato.ac.nz). See it in '/moa/classifiers'.

```
OISVM and IGNGSVM have previously been compared in http://hdl.handle.net/10016/19258.
Both implementations use [LibSVM for WEKA](https://weka.wikispaces.com/LibSVM?responseToken=128fd0dcc6e4ee7d48ef9847510ab2ed) (its jar file is present in the root of this repo). 
Support Vectors are retrieved from the SVM using the jar's object.
```

Wilson Edited (Also known as Edited Nearest-Neighbor)
===
Noise reduction algorithm based in Nearest Neighbors. Proposed by [Dennis L. Wilson in 1972](https://doi.org/10.1109/TSMC.1972.4309137). See it in '/weka/filters'.

GNG
===
Growing Neural Gas is implemented in [MOA](http://moa.cms.waikato.ac.nz) as topology generation method for IGNGSVM. Proposed by [Fritzke in 1995](https://papers.nips.cc/paper/893-a-growing-neural-gas-network-learns-topologies.pdf).

Task & Evaluation
====
EvaluateChunksTwoFiles, in '/moa/tasks',  evaluates test and train in parallel to allow online evaluation for non-stationary environments in presence of concept drift. Albeit this is not the only Task that can be run in MOA for this purpose. For example, run_example.sh uses [EvaluateInterleavedTestThenTrain](http://www.cs.waikato.ac.nz/~abifet/MOA/API/classmoa_1_1tasks_1_1_evaluate_interleaved_test_then_train.html) as task.
