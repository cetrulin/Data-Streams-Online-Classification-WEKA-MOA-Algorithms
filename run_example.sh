# Author Andres L. Suarez-Cetrulo (@cetrulin)
# WEKA and MOA used versions are the latest arc packages available in March 2012.

#!/bin/bash

EXTRAPATH="./moa/classes:./libsvm.jar:./weka/classes"
RESULTPATH="/Users/user_example/Workspace/WEKA&MOA/data"
WSIZE=2500
PROBLEM="HIGGS"
TSs=(200000)
neighbours=(3 5 7)
modes=(1 0)

IGNGSVM='igngsvm.IGNGSVM'
ALGORITHM=${IGNGSVM}
ALGOTOK="IGNGSVM"

for mode  in "${modes[@]}"; 
	do for neighbour in "${neighbours[@]}"; 
		do for TS in "${TSs[@]}"; 
			do java -cp moa.jar:weka.jar:$EXTRAPATH -javaagent:sizeofag.jar moa.DoTask \
			"EvaluateInterleavedTestThenTrain \
			 -l (igngsvm.IGNGSVM -t ${TS} -e ${mode} -Q 0.05 -n ${neighbour}  -b (weka.classifiers.functions.LibSVM -S 0 -K 0 -D 1 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 8.0 -E 0.0010 -P 0.1 -model /Users/user_example/Workspace/WEKA&MOA)) \
			 -s (ArffFileStream -f /Users/user_example/Desktop/datasets/Higgs/HIGGS.csv -c 1) \
			 -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
			 -f ${TS}" \
			> ${RESULTPATH}/${PROBLEM}_${ALGOTOK}_TS${TS}_vec${neighbour}_mode${mode}.txt 
		done
	done
done

# source getresults.sh