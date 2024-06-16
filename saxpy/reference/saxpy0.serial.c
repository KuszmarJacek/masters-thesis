#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float *x, *y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));


    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    saxpy(N, 2, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    free(x);
    free(y);

    return 0;
}