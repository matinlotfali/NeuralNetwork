using System;

using NeuralNetwork.Utilities;

namespace NeuralNetwork
{
    enum TransferFunction
    {
        Hard_Limit,
        Symmetrical_Hard_Limit,
        Linear,
        Saturating_Linear,
        Symmetric_Saturating_Linear,
        Log_Sigmoid,
        Hyperbolic_Tangent_Sigmoid,
        Positive_Linear,
        Competitive,
    }

    struct Layer
    {
        public int cellCount;
        public TransferFunction function;
        public double[,] weights;
        public double offsetFunction;

        public double[] Calculate(double[] input)
        {
            double[] output = MyMath.Multiply(input, weights);

            double max = 0;
            if (function == TransferFunction.Competitive)
                max = MyMath.GetMax(output);

            for (int i = 0; i < cellCount; i++)
            {
                double n = output[i];
                double a;
                switch (function)
                {
                    case TransferFunction.Hard_Limit:
                        {
                            a = (n + offsetFunction >= 0) ? 1 : 0;
                        }
                        break;

                    case TransferFunction.Symmetrical_Hard_Limit:
                        {
                            a = (n + offsetFunction >= 0) ? 1 : -1;
                        }
                        break;

                    case TransferFunction.Linear:
                        {
                            a = n + offsetFunction;
                        }
                        break;

                    case TransferFunction.Saturating_Linear:
                        {
                            n += offsetFunction;
                            if (n > 1)
                                a = 1;
                            else if (n < 0)
                                a = 0;
                            else
                                a = n;
                        }
                        break;

                    case TransferFunction.Symmetric_Saturating_Linear:
                        {
                            n += offsetFunction;
                            if (n > 1)
                                a = 1;
                            else if (n < -1)
                                a = 0;
                            else
                                a = n;
                        }
                        break;

                    case TransferFunction.Log_Sigmoid:
                        {
                            n += offsetFunction;
                            double e = Math.Exp(-n);
                            a = 1.0 / (1 + e);
                        }
                        break;

                    case TransferFunction.Hyperbolic_Tangent_Sigmoid:
                        {
                            n += offsetFunction;
                            double e1 = Math.Exp(n);
                            double e2 = Math.Exp(-n);
                            a = (e1 - e2) / (e1 + e2);
                        }
                        break;

                    case TransferFunction.Positive_Linear:
                        {
                            n += offsetFunction;
                            a = (n < 0) ? 0 : n;
                        }
                        break;

                    case TransferFunction.Competitive:
                        {
                            a = (n == max) ? 1 : 0;
                        }
                        break;
                }
            }
            return output;
        }
    }
}
