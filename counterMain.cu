#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>

#include "dataTransfer.cu"
#include "counter.cu"

int main (int argc, char *argv[]) {
  
  if (argc != 3) {
    printf ("Argument Error:  correct usage is\t./counter inputFile querySequence\n");
    return 1;
  }




  return 0;
}
