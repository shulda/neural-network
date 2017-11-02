0. This is my compulsory school software project. Its goal was to implement a simple neural network from scratch in C++, demonstrate that it works on the MNIST handwritten digit recognition problem and then try to use existing dataset (https://github.com/primaryobjects/voice-gender) to try to determine gender based on a voice sample. But for that case it turned out that my C++ sound processor cannot produce the same results as an R script that the dataset's author used, hence I couldn't used a network trained on the dataset to classify new samples.

1. Description of the project:
The goal of this project was to implement a sigmoid neural network with the backpropagation teaching algorithm and use this library to create a program recognizing the gender based on a sample of a voice recording.


2. Description of the files
This archive contains the following files:
- neural_network.hpp, an implementation of a three-layer sigmoid neural network. It can save itself to a file, load the file and feed forward given input.
- gradient_descent.hpp, an implementation of the gradient descent teaching algorithm. As one of its template parameters it takes a neural network which it is supposed to teach.
- mnist.[hc]pp, an application of the neural network library to solve handwritten digit recognition based on the MNIST data set (a standard task which I used for testing, calibration and comparison of my network with others - without much parameter tuning the network achieves about 97.5 % accuracy on test data).
- voice_recognition_net.[hc]pp, an application of the neural network library to try to recognize gender base on some spectral properties of a voice sample. 
- voice_processor.[hc]pp, a utility class for extracting the spectral classification properties from a raw voice sample.
- main.cpp, implementing the main function, contains simple demonstration of both digit recognition and voice recognition
- this README file
- several more files used needed for the demonstration


3. Required libraries and C++ version
- Because the whole solution is very linear algebra heavy, I decided to use a linear algebra C++ library, namely Armadillo: http://arma.sourceforge.net/ . It has its own dependencies well documented on the website (and binaries should be included in the download Armadillo package). I have not yet tested the project on Windows in Visual Studio
- It needs C++14 because of some auto in lambda syntax sugar. It should be easy to transform it to only require C++11
- For compilation with g++, the -larmadillo flag needs to be added !!at the end of the command!! (I don't understand why):
	g++ -std=c++14 -Wall -O3 -o rocnikac *.cpp -larmadillo


4. Required data sets
- Due to file size, I did not include the MNIST handwritten digits data set. Please download them from here http://yann.lecun.com/exdb/mnist/ (all four files (train|t10k)-(images|labels)-idx[13]-ubyte.gz) and extract them to a mnist/ folder.
- The file voice_gender_data contains preprocessed and shuffled training data for voice gender recognition obtained from https://github.com/primaryobjects/voice-gender


5. Comments for individual components
5.1 neural_network.hpp
The sizes of the layers are given as template parameters. The implementation does only assume the existence of one imput layer, one output layer and that between the layers are complete bipartite directed graphs. (Specifically it does not asume anything about the number of hidden layers.) Teh API currently supports only three-layer networks, which I chose for simplicity and because gradient descent algorithm cannot effectively teach multi-layer networks.
Supports saving weights to file.
Uses sigmoid neurons.

5.2 gradient_descent.hpp
Another templated class which provides the gradient descent teaching algorithm. It takes two template parameters - an instance of the NeuralNetwork template and a policy class providing some parameters for the teaching algorithm.

5.3 mnist.[hc]pp
A demonstration of the neural network and gradient descent implementations on standard data. As it is just a demonstration, it doesn't provide any API, it just runs the gradient descent algorithm in its constructor. Poor man's way to provide API would be to make the GradientDescent class public (hence also the NeuralNetwork class public), but wraping that up with some direct API is just a matter of a little bit straightforward work if someone wanted to use it to really clasify handwritten digits.
Without much parameter optimisation, the implementation achieved about 97.5% accuracy on an independent test data set.

5.4 voice_recognition_net.[hc]pp
Uses the gradient descent library, teaches it from given data (voice_gender_data), supports saving and loading and of course identifying the gender based on given classification parameters.
The spectral data have very different magnitudes, while the networks need each input to be roughly from the interval [0,1]. Because of that, mean and standard deviation of each input parameter are computed from the teaching data and then are used to normalize all inputs.
Using all 20 the network achieves 97% accuracy.

5.5 voice_processor.[hc]pp
Takes a raw 16-bit LPCM audio file in the correct endianity, encoded in signed integers with one channel as input, domputes its Fourier transform and from that it extracts several (12) spectral properties which can later be used as input for VoiceRecognitionNet.
On Linux, a correct raw audio file can be produced from a WAV file by the following command: 
	sox voice.wav --bits 16 --encoding signed-integer --endian little -c1 voice.raw
There are a couple of raw filed (together with their WAV counterparts) available in the archive.
There is an important problem: See section 6.


6. Why does the voice recognition not work?
The data I am using to teach the voice recognition network are preprocessed by an R program (see https://github.com/primaryobjects/voice-gender/blob/master/sound.R ). It basically calls the R warbleR package, which uses other package to process an audio signal and output 20 parameters describing its spectral properties.

It turns out that the R package does a big amount of work processing the audio, by which I mean that it tries to improve the data before running FFT (such as using voice reduction, different sampling window etc.). I cannot replicate this in C++ for two reasons, one is that it is difficult to find out what precisely is going on and the other is that it is relatively difficult area, I don't want to code something I do not understand the Mathematics behind and it would be a lot of code out of the intended scope of this project.

My C++ voice processor gives relatively similar basic spectral properties (mean frequency, median frequency etc.), which suggests that the processing is done correctly, but the higher order properties, such as kurtosis, differ very significantly (i.e. my program gives something absolutely different than the R program), which I suspect is the result of preprocessing (as it affects mostly higher order properties of the distribution). I tried to copy exactly the kurtosis (etc.) computation as the R package uses, but it had no effect.

This means that while the network performs very well on the provided data, it fails to clasify a given voice sample. The only way to fix this situation would be to get teaching data compatible with my voice processor, which means collecting all the raw audio data that the author of the dataset had collected and producing my own dataset. It should be doable in a couple-of-days-time, but to me it is not an interesting nor a learning experience.

Because of this I did not create any (command line) user interface.

Remark: If I had the audio data, I slightly suspect that the network should give good results even if just given the frequency distribution (on the human voice range, i.e. 0..280 Hy), so that would be something worth testing.

Note: The audio files 220.wav etc. contain a single sine wave and have been used to check whether the voice processor gives reasonable results.
