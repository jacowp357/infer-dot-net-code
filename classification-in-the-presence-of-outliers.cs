using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TestInfer
{
    class Program
    {
        static void Main(string[] args)
        {
            // x_0, x_1
            double[][] data = new double[40][] {
                        new double []   { 1.579866, 5.113884 },
                        new double []   { 2.596718, 5.731915 },
                        new double []   { 2.250065, 5.431592 },
                        new double []   { 1.670613, 4.528363 },
                        new double []   { 1.819446, 4.941831 },
                        new double []   { 2.662797, 5.065860 },
                        new double []   { 1.999470, 5.029127 },
                        new double []   { 0.871543, 4.512580 },
                        new double []   { 1.751953, 4.926783 },
                        new double []   { 2.765513, 5.743687 },
                        new double []   { 1.855384, 4.789793 },
                        new double []   { 3.096498, 6.281974 },
                        new double []   { 2.071130, 4.893247 },
                        new double []   { 1.920525, 4.957070 },
                        new double []   { 1.040348, 3.779229 },
                        new double []   { 2.128256, 4.971110 },
                        new double []   { 1.865504, 4.909423 },
                        new double []   { 2.214581, 5.388266 },
                        new double []   { 2.679222, 5.509920 },
                        new double []   { 2.511456, 5.758537 },
                        new double []   { 1.373862, 3.160759 },
                        new double []   { 1.089823, 3.321712 },
                        new double []   { 1.179304, 3.063109 },
                        new double []   { 1.222694, 3.319575 },
                        new double []   { 1.583625, 3.028068 },
                        new double []   { 1.813981, 2.564715 },
                        new double []   { 1.764641, 2.498540 },
                        new double []   { 1.942168, 2.343563 },
                        new double []   { 1.496726, 3.097716 },
                        new double []   { 1.248391, 3.171456 },
                        new double []   { 1.167128, 3.165683 },
                        new double []   { 1.776658, 2.666594 },
                        new double []   { 1.024596, 3.529081 },
                        new double []   { 1.495685, 3.055696 },
                        new double []   { 0.863904, 3.651468 },
                        new double []   { 1.814797, 2.607492 },
                        new double []   { 1.796724, 2.754365 },
                        new double []   { 0.927581, 3.627839 },
                        new double []   { 1.544352, 2.947286 },
                        new double []   { 1.400211, 3.180880} };

            Vector[] normalData = new Vector[20];
            Vector[] anomalousData = new Vector[20];

            for (int i = 0; i < 20; i++)
            {
                double[] input = new double[2] { data[i][0], data[i][1] };
                normalData[i] = Vector.FromArray(input);
            }

            for (int i = 0; i < 20; i++)
            {
                double[] input = new double[2] { data[i + 19][0], data[i + 19][1] };
                anomalousData[i] = Vector.FromArray(input);
            }

            Range k = new Range(2);

            VariableArray<Vector> means = Variable.Array<Vector>(k);
            means[k] = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(0.0, 0.0), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k);

            VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k);
            precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k);

            // define the Dirichlet prior over the normal data points (concentrate on class 0)
            Variable<Vector> weightsNormalPoints = Variable.Dirichlet(k, new double[] { 1, 0.1 });

            // define range for the normal data
            Range n = new Range(20);

            // define the x Gaussian random variables we will observe as normal data points
            VariableArray<Vector> xNormalPoints = Variable.Array<Vector>(n);

            // define latent z for all the normal data points
            VariableArray<int> zNormalPoints = Variable.Array<int>(n);

            using (Variable.ForEach(n))
            {
                zNormalPoints[n] = Variable.Discrete(weightsNormalPoints);

                using (Variable.Switch(zNormalPoints[n]))
                {
                    xNormalPoints[n] = Variable.VectorGaussianFromMeanAndPrecision(means[zNormalPoints[n]], precs[zNormalPoints[n]]);
                }
            }

            // define range for the anomalous data
            Range m = new Range(20);

            // define the x Gaussian random variables we will observe as anomalous data points
            VariableArray<Vector> xAnomalousPoints = Variable.Array<Vector>(m);

            // define the Dirichlet prior over the anomalous data points (concentrate on class 1)
            Variable<Vector> weightsAnomalousPoints = Variable.Dirichlet(k, new double[] { 0.1, 1 });

            // define latent z for all the anomalous data points
            VariableArray<int> zAnomalousPoints = Variable.Array<int>(m);

            using (Variable.ForEach(m))
            {
                zAnomalousPoints[m] = Variable.Discrete(weightsAnomalousPoints);

                using (Variable.Switch(zAnomalousPoints[m]))
                {
                    xAnomalousPoints[m] = Variable.VectorGaussianFromMeanAndPrecision(means[zAnomalousPoints[m]], precs[zAnomalousPoints[m]]);
                }
            }

            xNormalPoints.ObservedValue = normalData;
            xAnomalousPoints.ObservedValue = anomalousData;

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            //var weightsPost = ie.Infer<Dirichlet>(weightsNormalPoints);
            //var anomalyWeightsPost = ie.Infer<Dirichlet>(weightsAnomalousPoints);
            //var meansPost = ie.Infer<VectorGaussian[]>(means);
            //var precsPost = ie.Infer<Wishart[]>(precs);

            //Console.WriteLine("Dist over pi=" + weightsPost.GetMean());
            //Console.WriteLine("Dist over pi=" + anomalyWeightsPost.GetMean());
            //Console.WriteLine("Dist over means 0 =\n" + meansPost[0]);
            //Console.WriteLine("Dist over means 1 =\n" + meansPost[1]);
            //Console.WriteLine("Dist over precs 0 =\n" + precsPost[0].GetMean().Inverse());
            //Console.WriteLine("Dist over precs 1 =\n" + precsPost[1].GetMean().Inverse());


            // now let's try making a prediction with an "unseen" data point x
            Range xn = new Range(1);

            VariableArray<Vector> xNew = Variable.Array<Vector>(xn);

            VariableArray<int> zNew = Variable.Array<int>(xn);

            // define a uniform Dirichlet prior over the latent z
            Variable<Vector> zNewWeights = Variable.Dirichlet(k, new double[] { 1, 1 });

            using (Variable.ForEach(xn))
            {
                zNew[xn] = Variable.Discrete(zNewWeights);

                using (Variable.Switch(zNew[xn]))
                {
                    xNew[xn] = Variable.VectorGaussianFromMeanAndPrecision(means[zNew[xn]], precs[zNew[xn]]);
                }
            }

            Vector[] xNewData = new Vector[1];

            for (int i = 0; i < 1; i++)
            {
                double[] input = new double[2] { 3, 6 };
                xNewData[i] = Vector.FromArray(input);
            }

            xNew.ObservedValue = xNewData;

            var zNewPost = ie.Infer<Discrete[]>(zNew);

            Console.WriteLine(zNewPost[0]);
        }
    }
}
