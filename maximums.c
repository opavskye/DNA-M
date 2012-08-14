#include <string.h>
#include "constants.h"

int maximum (uint * list, int listLength) {
  
  int max = 0;

  for (int i = 1; i < listLength; i++)
    if (list[i] > list[max])
      max = i;

  return max;
}


bucketData summarizeMaximums (bucketData * data, int dataCount, int numMaxes) {

  bucketData max;
  
  // first value in each row of indices is index of which bucketaData is a max
  // second value is which item in that bucketData is a max
  int maxes[numMaxes][2];

  // initialize values
  for (int i = 0; i < numMaxes; i++) {
    maxes[i][0] = 0;
    maxes[i][1] = i;
  }
  
  // this won't work if the numMaxes buckets of the first data item have highest counts of everything
  for (int i = 1; i < dataCount; i++) { // loop to go through all data
    for (int j = 0; j < numMaxes; j++) { // loop to go through all elements in each dataBucket in data
      for (int k = 0; k < numMaxes; k++) { // loop to go through all elements in max
        if (data[i].count[j] > data[maxes[k][0]].count[maxes[k][1]]) {
          
          // make sure data[i].bucketContents[j] doesn't match any of the bucketContents in maxes
          int match = 1;
          for (int x = 0; x < numMaxes; x++)
            match = match && strcmp (data[i].bucketContents[j], 
                                     data[maxes[x][0]].bucketContents[maxes[x][1]]);

          if (match) {

            // store old item at this location
            int temp[2] = {maxes[k][0], maxes[k][1]};

            // put new value into maxes
            maxes[k][0] = i;
            maxes[k][1] = j;

            // shift other maxes down
            for (int x = numMaxes - 1; x > k; x--) {
              maxes[x][0] = maxes[x - 1][0];
              maxes[x][1] = maxes[x - 1][1];
            }
          
            // put old maximum at this k index in its new location, if it's still viable
            if (k + 1 < numMaxes) {
              maxes[k + 1][0] = temp[0];
              maxes[k + 1][1] = temp[1];
            }

            // don't put this data[i].*[j] element into anywhere else in the maxes
            break;
          }
        }
      }
    }
  }

  // copy maxes indices of data into max
  for (int i = 0; i < numMaxes; i++) {
    max.sequenceIndex[i] = data[maxes[i][0]].sequenceIndex[maxes[i][1]];
    max.bucketNum[i] = data[maxes[i][0]].bucketNum[maxes[i][1]];
    for (int j = 0; j < 21; j++)
      max.bucketContents[i][j] = data[maxes[i][0]].bucketContents[maxes[i][1]][j];
    max.count[i] = data[maxes[i][0]].count[maxes[i][1]];
  }
  
  return max;
}

void findMaximums (uint * bucketCounts, int numBuckets, int * maxes, int numMaxes) {

  for (int i = 0; i < numMaxes; i++)
    maxes[i] = 0;

  for (int i = 0; i < numBuckets; i++)
    for (int j = 0; j < numMaxes; j++)
      if (bucketCounts[i] > bucketCounts[maxes[j]]) {
      
        // store old item at this location
        int temp = maxes[j];

        // put new value into maxes
        maxes[j] = i;

        // shift other maxes down
        for (int k = numMaxes - 1; k > j; k--) 
          maxes[k] = maxes[k-1];
        
        // put old maximum at this j index in its new location, if it's still viable
        if (j + 1 < numMaxes)
          maxes[j + 1] = temp;

        // don't put this bucketCounts element into anywhere else in the maxes
        break;
      }
}
