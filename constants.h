/* Copyright 2012 by Erik Opavsky
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define THREADS_PER_BLOCK 1024
#define OUTPUTS_TO_KEEP 100

typedef struct {
  int sequenceIndex[OUTPUTS_TO_KEEP];
  int bucketNum[OUTPUTS_TO_KEEP];
  char bucketContents[OUTPUTS_TO_KEEP][21];
  int count[OUTPUTS_TO_KEEP];
} bucketData;
