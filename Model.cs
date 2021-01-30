using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace ForwardFeed
{
    [Serializable]
    public class Model
    {
        private Matrix<double>[] _weights;
        private Vector<double>[] _bias;
        private int[] _size;

        public Matrix<double>[] Weights => _weights;
        public Vector<double>[] Bias => _bias;
        public int[] Size => _size;

        public Model(int[] size)
        {
            _size = size;
            _weights = new Matrix<double>[Size.Length - 1];
            _bias = new Vector<double>[Size.Length - 1];

            for (int i = 0; i < Size.Length - 1; i++)
            {
                _weights[i] = Matrix<double>.Build.Dense(Size[i], Size[i + 1], 0);
                _bias[i] = Vector<double>.Build.Dense(Size[i + 1], 0);
            }
        }
        public Model((double[,] Weights, double[] Bias)[] datas)
        {
            _size = new int[datas.Length + 1];
            _weights = new Matrix<double>[datas.Length];
            _bias = new Vector<double>[datas.Length];

            for (int i = 0; i < datas.Length; i++)
            {
                _weights[i] = Matrix<double>.Build.DenseOfArray(datas[i].Weights);
                _bias[i] = Vector<double>.Build.DenseOfArray(datas[i].Bias);
                _size[i + 1] = Bias[i].Count;
            }

            _size[0] = Weights[0].RowCount;
        }

        private Model Scale(double scalar)
        {
            Model result = new Model(Size);
            Parallel.For(0, Size.Length - 1, i =>
            {
                result.Weights[i] = Weights[i] * scalar;
                result.Bias[i] = Bias[i] * scalar;
            });
            return result;
        }
        private Model Add(Model adder)
        {
            Model result = new Model(Size);
            Parallel.For(0, Size.Length - 1, i =>
            {
                result.Weights[i] = Weights[i] + adder.Weights[i];
                result.Bias[i] = Bias[i] + adder.Bias[i];
            });
            return result;
        }
        private double Sigmoid(double value)
        {
            double k = Math.Exp(value);
            return k / (1.0 + k);
        }
        private double Dsigmoid(double k)
        {
            return k * (1.0 - k);
        }

        public void Randomize(int seed)
        {
            for (int i = 0; i < Size.Length - 1; i++)
            {
                Weights[i] = Matrix<double>.Build.Random(Weights[i].RowCount, Weights[i].ColumnCount, seed);
                Bias[i] = Vector<double>.Build.Random(Bias[i].Count, seed);
            }
        }
        public Vector<double>[] FeedForward(Vector<double> input)
        {
            Vector<double>[] result = new Vector<double>[Size.Length];
            result[0] = input;
            for (int i = 0; i < Size.Length - 1; i++)
                result[i + 1] = (result[i] * Weights[i] + Bias[i]).Map(Sigmoid);

            return result;
        }
        public Vector<double>[] GetErrors(Vector<double> guess, Vector<double> answer)
        {
            Vector<double>[] errors = new Vector<double>[Size.Length];
            errors[errors.Length - 1] = guess - answer;

            for (int i = errors.Length - 1; i > 1; i--)
                errors[i - 1] = errors[i] * Weights[i - 1].Transpose();

            Extensions.AddScore(errors.Last().ToArray());

            return errors;
        }
        public Model GetGradient((Vector<double> Input, Vector<double> Answer) data)
        {
            Model result = new Model(Size);
            Vector<double>[] guess = FeedForward(data.Input);
            Vector<double>[] errors = GetErrors(guess.Last(), data.Answer);

            for (int i = Size.Length - 1; i > 0; i--)
            {
                result.Bias[i - 1] = guess[i].Map(Dsigmoid).PointwiseMultiply(errors[i]);
                result.Weights[i - 1] = guess[i - 1].ToColumnMatrix() * result.Bias[i - 1].ToRowMatrix();
            }

            return result;
        }
        public Model GetBatchGradient((Vector<double> Input, Vector<double> Answer)[] datas)
        {
            Model result = new Model(Size);

            for (int i = 0; i < datas.Length; i++)
            {
                result += GetGradient(datas[i]);
            }

            return result / datas.Length;
        }

        public (double[,], double[])[] GetCheckPoints()
        {
            (double[,] Weights, double[] Bias)[] result = new (double[,], double[])[Size.Length - 1];
            for (int i = 0; i < Size.Length - 1; i++)
            {
                result[i].Weights = Weights[i].ToArray();
                result[i].Bias = Bias[i].ToArray();
            }
            return result;
        }

        public static Model operator *(Model left, double right)
        {
            return left.Scale(right);
        }
        public static Model operator /(Model left, double right)
        {
            return left.Scale(1 / right);
        }
        public static Model operator +(Model left, Model right)
        {
            return left.Add(right);
        }
        public static Model operator -(Model left, Model right)
        {
            return left.Add(right.Scale(-1));
        }
    }
}
