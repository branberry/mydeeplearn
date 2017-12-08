import 'babel-polyfill';
import Chart from 'chart.js';
//import './religiondata/global.csv';
import {Array1D, Array2D, CostReduction, Graph, InCPUMemoryShuffledInputProviderBuilder, NDArray, NDArrayMathGPU, Scalar, Session, SGDOptimizer, Tensor} from 'deeplearn';

/** Runs the example. */
async function runExample() {
  const graph = new Graph();
  // Make a new input in the graph with the shape [] for a scalar
  const x = graph.placeholder('x',[]);

  // random scalar values for the Graph
  const a = graph.placeholder('a',[]);
  const b = graph.placeholder('b',[]);
  const c = graph.placeholder('c',[]);
  // creating new tensors to represent the output the operations of the quadratic
  const order2 = graph.multiply(a,graph.square(x));
  const order1 = graph.multiply(b,x);
  const y = graph.add(graph.add(order2,order1), c);
  // for training we need a label and a cost function
  const yLabel = graph.placeholder('y label',[]);
  // providing a mean squared cost function.  cost = (y-ylabel)^2
  const cost = graph.meanSquaredCost(y,yLabel);

  // the graph is set up, now we must evaluate it
  // the session object is needed to evaluate the graph
  const math = new NDArrayMathGPU();
  const session = new Session(graph,math);
  await math.scope(async (keep,track) =>{

  });

}

runExample();
