using System;

using NeuralNetwork.Utilities;

namespace NeuralNetwork
{
    class MainClass
    {

        public static void Main(string[] args)
        {
            //// inputs            
            Byte[] inputByte = MyMath.GetRandom(2000);

            int layersCount = 2;
            Layer[] layers = new Layer[layersCount];            
            layers[0] = new Layer()
            {
                cellCount = 300,
                function = TransferFunction.Linear,
                offsetFunction = 0.5,
                weights = MyMath.GetRandom(inputByte.Length, 300)
            };
            layers[1] = new Layer()
            {
                cellCount = 15,
                function = TransferFunction.Competitive,
                offsetFunction = 0,
                weights = MyMath.GetRandom(300, 150)
            };

            ///start
            double[] input = new double[inputByte.Length];
            for (int i = 0; i < input.Length; i++)
                input[i] = inputByte[i] / 255;

            double[] values = input;
            for (int i = 0; i < layers.Length; i++)
                values = layers[i].Calculate(values);


            //Console.ReadKey();
        }
    }
}
