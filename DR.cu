#include <cudnn.h>
#include <assert.h>
#include <cuda_runtime_api.h>
//#include <esbmc_atomic.h>

volatile float shared_out = 0;  // Shared output variables

void simulateConvolutionOperation(float modify) {
    // The following actions simulate changes to shared variables
    __ESBMC_atomic_begin();
    shared_out += modify;  // Modelling of competitive conditions
    __ESBMC_atomic_end();
}

int main() {
     // Modelling two possible execution paths
    if (nondet_bool()) {
        simulateConvolutionOperation(10);
    } else {
        simulateConvolutionOperation(-10);
    }

    // Verify the value of shared_out
    assert(shared_out == 0.0f); 

    return 0;
}
