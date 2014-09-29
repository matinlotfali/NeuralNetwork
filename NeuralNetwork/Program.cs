using System;

using NeuralNetwork.Utilities;

namespace NeuralNetwork
{
    class MainClass
    {

        public static void Main(string[] args)
        {
            //// inputs            
            Byte[] inputByte = MyMath.GetRandomBytes(2);

            double n = 0.1;
            int layersCount = 2;
            Layer[] layers = new Layer[layersCount + 1];

            layers[0] = new Layer();
            layers[0].cellCount = inputByte.Length;
            layers[0].output = new double[inputByte.Length];
            for (int i = 0; i < inputByte.Length; i++)
                layers[0].output[i] = inputByte[i] / 255.0;

            layers[1] = new Layer();
            layers[1].cellCount = 3;
            layers[1].function = TransferFunction.Linear;
            layers[1].bias = MyMath.GetRandomDoubles(layers[1].cellCount);
            layers[1].weights = MyMath.GetRandom(layers[0].cellCount, layers[1].cellCount);

            layers[2] = new Layer();
            layers[2].cellCount = 1;
            layers[2].function = TransferFunction.Linear;
            layers[2].bias = MyMath.GetRandomDoubles(layers[2].cellCount);
            layers[2].weights = MyMath.GetRandom(layers[1].cellCount, layers[2].cellCount);            

            for (int z = 0; z < 50; z++)
            {
                //feed forward all            
                for (int i = 1; i < layers.Length; i++)
                    layers[i].FeedForward(layers[i - 1].output);

                //backpropagate last layer
                int l = layers.Length - 1;

                double[] targetOutput = new double[layers[l].cellCount];

                //calculate errors
                layers[l].errors = new double[layers[l].cellCount];
                for (int i = 0; i < targetOutput.Length; i++)
                    layers[l].errors[i] = layers[l].output[i] - targetOutput[i];

                Console.WriteLine("FeedForward error: " + Math.Round(MyMath.Sum(layers[l].errors),4));

                //calculate weights
                ChangeWeights(n, layers, l);

                //backpropagate other layers
                for (l = layers.Length - 2; l > 0; l--)
                {
                    //calculate errors
                    layers[l].errors = new double[layers[l].cellCount];
                    for (int i = 0; i < layers[l].cellCount; i++)
                        for (int j = 0; j < layers[l + 1].cellCount; j++)
                            layers[l].errors[i] = layers[l + 1].errors[j] * layers[l + 1].weights[i, j];

                    //calculate weights
                    ChangeWeights(n, layers, l);
                }
            }

            Console.ReadKey();
        }

        private static void ChangeWeights(double n, Layer[] layers, int l)
        {
            double dw;
            for (int i = 0; i < layers[l].cellCount; i++)
            {
                for (int j = 0; j < layers[l - 1].cellCount; j++)
                {
                    dw = -n * layers[l].errors[i] * layers[l - 1].output[j];
                    layers[l].weights[j, i] += dw;
                }
                dw = -n * layers[l].errors[i];
                layers[l].bias[i] += dw;
            }
        }
    }
}
