using System;

using NeuralNetwork.Utilities;

namespace NeuralNetwork
{
    class MainClass
    {

        public static void Main(string[] args)
        {
            //// inputs            
            Byte[] inputByte = MyMath.GetRandom(2);

            double n = 0.1;
            int layersCount = 2;
            Layer[] layers = new Layer[layersCount];
            layers[0] = new Layer()
            {
                cellCount = 3,
                function = TransferFunction.Linear,
                offsetOfFunction = 0.5,
                weights = MyMath.GetRandom(inputByte.Length, 3)
            };
            layers[1] = new Layer()
            {
                cellCount = 1,
                function = TransferFunction.Linear,
                offsetOfFunction = 0,
                weights = MyMath.GetRandom(layers[0].cellCount, 1)
            };

            ///start
            double[] input = new double[inputByte.Length];
            for (int i = 0; i < input.Length; i++)
                input[i] = inputByte[i] / 255.0;

            //feed forward all
            layers[0].FeedForward(input);
            for (int i = 1; i < layers.Length; i++)
                layers[i].FeedForward(layers[i - 1].output);

            //backpropagate last layer
            int l = layers.Length - 1;

            double[] targetOutput = new double[layers[l].cellCount];

            //calculate errors
            layers[l].errors = new double[layers[l].cellCount];
            for (int i = 0; i < targetOutput.Length; i++)
                layers[l].errors[i] = layers[l].output[i] - targetOutput[i];

            //calculate weights
            ChangeWeights(n, layers, l);

            //backpropagate other layers
            for (l = layers.Length - 2; l >= 0; l--)
            {
                //calculate errors
                layers[l].errors = new double[layers[l].cellCount];
                for (int i = 0; i < layers[l].cellCount; i++)
                    for (int j = 0; j < layers[l + 1].cellCount; j++)
                        layers[l].errors[i] = layers[l + 1].errors[j] * layers[l + 1].weights[i, j];

                //calculate weights
                ChangeWeights(n, layers, l + 1);
            }

            //Console.ReadKey();
        }

        private static void ChangeWeights(double n, Layer[] layers, int l)
        {
            for (int j = 0; j < layers[l - 1].cellCount; j++)
                for (int i = 0; i < layers[l].cellCount; i++)
                {
                    double dw = -n * layers[l].errors[i] * layers[l - 1].output[j];
                    layers[l].weights[j, i] += dw;
                }
        }
    }
}
