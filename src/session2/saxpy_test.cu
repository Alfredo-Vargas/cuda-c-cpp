#include <stdio.h>

#define N (2048 * 2048)

__global__
void saxpy(float * a, float * b, float * c, const int N)
{
  a[0] = 1;
}


int main() {
  return 0;
}
