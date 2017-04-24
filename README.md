# Scalable nonparametric Bayesian multilevel clustering
### Source codes for stochastic variational inference for nonparametric Bayesian multilevel clustering models (MC2SVI) [Java/ Apache Spark]
This package implements the **stochastic variational inference for nonparametric Bayesian multilevel clustering models**  (MC2SVI) described in the following paper:

Huynh, Viet, Phung, Dinh, Venkatesh, Svetha, Nguyen, Xuan Long, Hoffman, Matt and Bui, Hung Hai 2016, Scalable nonparametric Bayesian multilevel clustering, in UAI 2016: Proceedings of the 32nd Conference on Uncertainty in Artificial Intelligence, AUAI Press, Corvallis, Or., pp. 289-298. 

**Disclaimer**: We have made our best effort in ensuring fairness in acknowledging existing codes and any materials we used. However, if you have any question/concern, please write to us.

# Using the code 
## Data
Each dataset includes **three** data files: content, context and meta data. In data folder, a sample of dataset, NIPS is included:
+ **content_nips.txt**: the content file which contains spare vector in libsvm format
+ **context_nips.txt**: the context file spare vector in libsvm format
+ **meta_nips.txt**: describe the dimensions of content and context data

## Configuration file: *config.properties*
+ mc2.trunM=150 	% truncation level for number of topics
+ mc2.trunK=80  	% truncation level for number of clusters
+ mc2.trunT=100 	% truncation level for number of topics for each cluster (restaurant)
+ mc2.aa = 10 	% concentration for cluster proportion
+ mc2.ee = 10 	% concentration for topic proportion at restaurant level
+ mc2.vv = 10 	% concentration for topic proportion d
+ mc2.batchSize=100 	%mini-batch size
+ mc2.numIter=1  	% number of running epochs
+ mc2.varrho = 1 	% learning rate 
+ mc2.iota = 0.8 	% learning rate

+ mc2.contentDirichletSym=0.001 	% prior parameter for content
+ mc2.contextDirichletSym=0.1 		% prior parameter for context
+ mc2.contextType=Multinomial 		% context distribution type

+ mc2.metaPath=meta_nips.txt		% path to meta data file 
+ mc2.contentPath=content_nips.txt	% path to content data file
+ mc2.contextPath=context_nips.txt	% path to context data file
+ mc2.outFolderPath=out		% path to output folder

### Install Apache Spark on the local machine
+	Installation: Download Spark 1.5.1 from http://spark.apache.org/downloads.html (spark-1.5.1-bin-hadoop2.6.tgz)  unzip to folder spark-1.5.1-bin-hadoop2.6
+	Set PATH to the folder spark-1.5.1-bin-hadoop2.6

### Running
+	Open command line ( terminal)
+	Change to code folder
+	Run: *spark-submit  --master local[8] BNPStat.jar config.properties*
+	Output will be store in mc2.outFolderPath

### Output (in matlab file format)
The variables stored in each matlab file after running each mini-batch
+ pp: the content atoms
+ qq: the context atoms 
+ qcc:  corresponding to μ^c
+ qzzs: corresponding to μ^z
+ rhos, varphis, zetas: stick breaking hyperparamters (corresponding to λ^β,λ^ϵ,λ^τ)



