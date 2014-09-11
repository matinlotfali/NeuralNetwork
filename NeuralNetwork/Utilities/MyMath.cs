using System;

namespace NeuralNetwork.Utilities
{
    class MyMath
    {
        static Random rand = new Random();
        public static double[,] GetRandom(int len1, int len2)
        {
            double[,] result = new double[len1, len2];
            for (int i = 0; i < len1; i++)
                for (int j = 0; j < len2; j++)
                    result[i, j] = rand.NextDouble();
            return result;
        }

        public static byte[] GetRandom(int len)
        {
            byte[] result = new byte[len];
            rand.NextBytes(result);
            return result;
        }

        public static double GetMax(double[] a)
        {
            double max = a[0];
            int l = a.Length;

            unsafe
            {
                fixed (double* pa = a)
                {
                    for (int i = 1; i < l; i++)
                    {
                        double token = pa[i];
                        if (token > max)
                            max = token;
                    }
                }
            }

            return max;
        }

        public static double[] Multiply(double[] a, double[,] m)
        {
            int cellCount = m.GetLength(1);
            int l = a.Length;
            double[] output = new double[cellCount];

            //for (int i = 0; i < cellCount; i++)
            //{
            //    double sum = 0;
            //    for (int j = 0; j < l; j++)
            //        sum += a[j] * m[j, i];
            //    output[i] = sum;
            //}

            unsafe
            {
                fixed (double* pm = output, pm1 = a, pm2 = m)
                {
                    for (int i = 0; i < cellCount; i++)
                    {
                        int i2 = i;
                        double sum = 0;
                        for (int j = 0; j < l; j++, i2 += cellCount)
                            sum += pm1[j] * pm2[i2];
                        pm[i] = sum;
                    }
                }
            }

            return output;
        }

    }
}
