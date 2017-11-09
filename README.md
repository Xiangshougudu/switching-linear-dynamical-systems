# How to Use

The main file "slds_max.py" implements the actual model and learning procedure. The model itself is run and evaluated using "Testsuite.py". In there, a couple of different test classes are implemented each providing succinct testing methods for each submodule (which can be enhanced or added as needed.)

For example, if we were to test a certain functionality of the Kalman Filter, we would place this into the corresponding class:

    class KalmanFilterTestCase(SetupTest):
    """
    Tests for the Kalman filter
        
    Functions
    -------
    """
    def test_whatever(self):
        # continue as you want


    
Overall, the goal is to infer modes of the given trajectory. This is being evaluated in the test class

   
    class BlockSamplerTestCase(SetupTest):
    """        
    Functions
    -------
    """
    ...
        
    test_z_sampling(self):      
        
  
Here, "test_z_sampling" repeatedly samples the mode sequence until the Gibbs sampler converges to the real posterior. The mode-specific dynamical parameters can be easily accessed through the SLDS object itself.

"Heuristic.py" provides a simple heuristic measure of finding the mode-switching points in the trajectory serving as a baseline. It should be noted that this approach is not associated with the SLDS methodology itself.

# Foundation & Theory
Refer to the pdf for a more succinct description of the entire process. The references mentioned in the pdf are also an excellent source for further reading in that regard.

