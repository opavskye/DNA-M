#define THREADS_PER_BLOCK 1024
#define OUTPUTS_TO_KEEP 10

typedef struct {
  int sequenceIndex[OUTPUTS_TO_KEEP];
  int bucketNum[OUTPUTS_TO_KEEP];
  char bucketContents[OUTPUTS_TO_KEEP][21];
  int count[OUTPUTS_TO_KEEP];
} bucketData;
