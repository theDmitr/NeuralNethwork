# Neural Network

This library will help you configure your neural networks, share them, as well as datasets for them.

# Examples
## Initialization
The neural network has two constructors, the second differs in that it accepts already defined weights.
> **Note:** The second is highly not recommended; it exists for initializing a neural network based on previously saved data, that is, for a saving system.
```java
NeuralNetwork(int[] layersNeuronsCounts, boolean hasBias, NeuronActivation neuronActivation)
NeuralNetwork(int[] layersNeuronsCounts, boolean hasBias, NeuronActivation neuronActivation, double[][][] weights)
```
To initialize, for example, a neural network with two input, two hidden and one output neurons, using `BIAS` neurons and the `SIGMOID` activation function, you only need:
```java
NeuralNetwork neuralNetwork = new NeuralNetwork(new int[] { 2, 2, 1 }, true, NeuronActivation.SIGMOID);
```
## Learning
### Datasets
For visual structuring, there is a separate class for data: `LearnDataset`.
The constructor accepts the number of input and output neurons, and can also accept a ready-made data set .
> **Note:** The second is again not recommended, this is done to save and load data.
```java
LearnDataset(int inputNeurons, int outputNeurons)
LearnDataset(int inputNeurons, int outputNeurons, List<double[]> set)
```
In order to create a dataset for training a neural network, you simply need to create an instance of the LearnDataset class.
```java
LearnDataset dataset = new LearnDataset(2, 1);
```
The `insert` method is used for filling; accepts a `double[]` array with a length equal to the sum of the number of input and output neurons.
Here, for example, is setting up a `dataset` for teaching logical OR:
```java
dataset.insert(new double[]{ 0, 0, 0 });
dataset.insert(new double[]{ 0, 1, 1 });
dataset.insert(new double[]{ 1, 0, 1 });
dataset.insert(new double[]{ 1, 1, 1 });
```
### Learning tools
Training can be carried out through a certain number of iterations or until the neural network error exceeds the transmitted value.
#### Iteration method
To learn through iterations, you should call the `learn` method on the neural network, passing there the `dataset`, `learningRate`, `moment` and `iterations`.
```java
float learningRate = 1.0f;  
float moment = 0.8f;  
int iterations = 10_000;  
neuralNetwork.learn(dataset, learningRate, moment, iterations);
```
After this, after a certain amount of time (depending on the training parameters and the size of the neural network), the method will complete its work and the neural network will be trained based on this data.
#### Max error method
To train using the maximum error, it is enough to pass the maximum error instead of iterations, followed by the maximum number of iterations and the method for calculating the error.
> **ErrorCalcType** is also defined within this library.
```java
float error = 0.01f;
int maxIterations = 10_000;
ErrorCalcType errorCalcType = ErrorCalcType.ROOT_MSE;
neuralNetwork.learn(dataset, learningRate, moment, error, maxIterations, errorCalcType);
```
## Predict
In order for the neural network to predict something based on the input data, you should call the only `predict` method for this purpose
As an argument, the method takes an `double[]` array for input data (the dimension of the array is equivalent to the number of input neurons of the neural network), and also returns an `double[]` array, only this is the result of the work of the neural network and its dimension is equivalent to the number of output neurons.
```java
double[] inputData = new double[] { 1, 0 };  
neuralNetwork.predict(inputData);
```
## Saving/Load NeuralNetwork and LearnDataset
An interface `IParser<T>` was created to save object data.
Interfaces contain these methods:
```java
void parseOut(T t, OutputStream stream);  
T parseIn(InputStream stream);
```
There are two parsers: `NeuralNetworkParser` and `LearnDatasetParser`, both of them implement this interface.
### NeuralNetwork
To write a `NeuralNetwork` somewhere (for example, to a file), the `parseOut` method is used, which accepts the `NeuralNetwork` itself and the output stream.
```java
File file = new File("NeuralNetwork");  
FileOutputStream outputStream = new FileOutputStream(file);  
NeuralNetworkParser parser = new NeuralNetworkParser();  
parser.parseOut(neuralNetwork, outputStream);
```
On the contrary, to load the `NeuralNetwork` we will use the `parseIn` method, passing the input stream into it. This method will return the resulting `NeuralNetwork`.
```java
FileInputStream inputStream = new FileInputStream(file);  
NeuralNetwork parsed = parser.parseIn(inputStream);
```
### LearnDataset
In principle, saving/loading a `LearnDataset` is identical.

Saving:
```java
File file = new File("LearnDataset");  
FileOutputStream outputStream = new FileOutputStream(file);  
LearnDatasetParser parser = new LearnDatasetParser();  
parser.parseOut(dataset, outputStream);
```
Load:
```java
FileInputStream inputStream = new FileInputStream(file);  
LearnDataset parsed = parser.parseIn(inputStream);
```
