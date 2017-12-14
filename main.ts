/*
 This file will be trained to predict complementary colors 
 */
import {Array1D, CostReduction, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, NDArrayMath, NDArrayMathGPU, Session, SGDOptimizer, Tensor} from 'deeplearn';

class ComplementaryColorModel {
    // runs training
    session: Session;

    //encapsulates math operations on the CPU and GPU.
    math: NDArrayMath = new NDArrayMathGPU();

    // An optimizer with a certain initial learning rate.  Used for training.
    intialLearningRate = 0.042;
    optimizer: SGDOptimizer;
}