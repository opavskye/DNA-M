  Copyright 2012 by Erik Opavsky

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.


This is software for the analysis of methylated DNA sequences.  The following 
programs are included:  sequencer, counter, promoterMerger, and ConsensusSearch.

Sequencer and counter can only be used on CUDA enabled machines, with a CUDA device 
of at least Compute Capability 2.0 or greater.  The purpose of the sequencer program 
is to find the most common motifs in a given set of DNA sequences, with the ability 
for the user to give a "match threshold" which qualifies sequence matching.  The 
purpose of the counter program is to count the number of occurances of a subsequence 
within a large set of sequences.  This also uses the "match threshold" to account 
for slight differences in sequence.  Neither program accounts for insertions or 
deletions in nucleotides, only mutation of nucleotides is accounted for by match 
threshold.  For example, a .8 match threshold when searching for subsequences of 
length 10 in the data requires 8 out of the 10 nucleotides to match.  There can 
only be one match for a subsequence per sequence in the data set, so the maximum 
number of matches found for a subsequences is the number of sequences there are in 
the data set.

Modification to these programs, in the files sequencerMain.cu, counterMain.cu, and 
dataTransfer.cu, is likely necessary to fit the programs to a new data set.  These 
modifications must take into account the length of each sequence to be analyzed, in 
the variable sequenceLength, the number of sequences to be analyzed, in the variable 
numSequences, and how the data is read in, in the function readSequences() in 
dataTransfer.cu.  Additionally, the program parameters minLength and maxLength in 
sequencerMain.cu should be taken into account when running sequencer, as those 
variables determine what range of motif lengths will be tested.

Usage for sequencer is:		./sequencer inputFile outputFile matchAccuracy
Usage for counter is:		./counter inputFile querySequence matchAccuracy
_____

The promoterMerger program is designed to merge data sets on promoter data based 
on the "Distance_TSS" column in a .csv file.  It will select the rows of each 
inputFile which have a Distance_TSS value between minBound and maxBound and put 
them in the outputFile.

Usage for promoterMerger is:	
	./promoterMerger minBound maxBound outputFile inputFile1 inputFile2 ...
_____	

The ConsensusSearch program looks over a set of methylation data and extracts rows 
from the inputFiles to the outputFile based on certain conditions.  Included are 
the .java files ConsensusSearch.java and HeaderInfo.java, needed to create a 
working java project for ConsensusSearch.
 
The first condition is that the Distance_TSS column value is positive.

The second condition is that there are at least 10 unique Center_CCGG_HG_19 values 
for any geneSymbol.  However, not just the unique Center_CCGG_HG_19 values are 
copied over to the output file, but the entire geneSymbol group is copied over for 
the values under which both the first condition holds true and the geneSymbol 
groups under which the second condition holds true.

In addition, when the values are copied over, there will be a new column created 
titled "length" which stores the value of txEnd - txStart.

An important condition which must hold true for this program to function correctly 
is that each data set is grouped by geneSymbol.

Usage for ConsensusSearch is:	
	./ConsensusSearch outputFile inputFile1 inputFile2 ...


These programs were orignally created for use by Opavsky Lab at the University of 
Nebraska Medical Center's Eppley Institute for Research in Cancer.  They are 
targeted for use with data in the format which is used there, and will require an 
experienced user to use with other data, even for very similar purposes.

For more information on these programs, and for help with using them, please contact 
opavskye@grinnell.edu with any questions.
