using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

namespace ForwardFeed
{
    public class NeuralNetwork
    {
        public Model CheckPoint;
        public double Alpha = 0.01;
        public double Beta1 = 0.9;
        public double Beta2 = 0.99;
        public double Gamma = 0.9;
        public int BatchSize = 128;

        public NeuralNetwork(params int[] size)
        {
            CheckPoint = new Model(size);
        }
        public NeuralNetwork(Model checkPoint)
        {
            CheckPoint = checkPoint;
        }

        public void Randomize(int seed = 69)
        {
            Extensions.Log("Randomizing weights and biases...");
            CheckPoint.Randomize(seed);
        }

        public (double, double) Train((double[] Input, double[] Answer)[] rawDatas, Optimizer optimizer)
        {
            Extensions.ResetScore(rawDatas.Length);
            var datas = MakeData(rawDatas);

            Console.WriteLine();

            bool test = optimizer switch
            {
                Optimizer.SGD => SGD(datas),
                Optimizer.MiniBatchSGD => MiniBatchSGD(datas),
                Optimizer.Nesterov => Nesterov(datas),
                _ => throw new ArgumentOutOfRangeException()
            };

            Extensions.Log("[debug] " + (test ? "success" : "something went wrong"));
            return Extensions.GetScore();
        }
        public (double, double) Evaluate((double[] Input, double[] Answer)[] rawDatas)
        {
            Extensions.ResetScore(rawDatas.Length);
            var datas = MakeData(rawDatas);

            Console.WriteLine();
            foreach((Vector<double> Input, Vector<double> Answer) data in datas)
            {
                CheckPoint.GetErrors(CheckPoint.FeedForward(data.Input).Last(), data.Answer);
                Extensions.ProgressBar();
            }

            return Extensions.GetScore();
        }

        private bool SGD((Vector<double>, Vector<double>)[] datas)
        {
            foreach (var data in datas)
            {
                CheckPoint -= CheckPoint.GetGradient(data) * Alpha;
                Extensions.ProgressBar();
            }

            return true;
        }
        private bool MiniBatchSGD((Vector<double>, Vector<double>)[] datas)
        {
            foreach (var batch in GetBatches(datas))
            {
                CheckPoint -= CheckPoint.GetBatchGradient(batch) * Alpha;
                Extensions.ProgressBar();
            }

            return true;
        }
        private bool Nesterov((Vector<double>, Vector<double>)[] datas)
        {
            Model velocity = new Model(CheckPoint.Size);
            Model gradient = new Model(CheckPoint.Size);

            foreach (var batch in GetBatches(datas))
            {
                gradient = (CheckPoint - velocity * Gamma).GetBatchGradient(batch);
                velocity = velocity * Gamma + gradient * Alpha;
                CheckPoint -= velocity / batch.Length;
                Extensions.ProgressBar();
            }

            return true;
        }

        private (Vector<double>, Vector<double>)[][] GetBatches((Vector<double>, Vector<double>)[] datas)
        {
            return datas.Select((value, index) => (value, index)).GroupBy(x => x.index / BatchSize).Select(x => x.Select(y => y.value).ToArray()).ToArray();
        }
        private (Vector<double>, Vector<double>)[] MakeData((double[] Input, double[] Output)[] datas)
        {
            return datas.Select(x => (Vector<double>.Build.DenseOfArray(x.Input), Vector<double>.Build.DenseOfArray(x.Output))).ToArray();
        }

        public enum Optimizer
        {
            SGD,
            MiniBatchSGD,
            Nesterov
        }
    }
}
