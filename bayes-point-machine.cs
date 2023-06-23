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
using System.IO;
using static System.Formats.Asn1.AsnWriter;

namespace TestInfer
{
    class Program
    {
        static void BayesPointMachine(Vector[] xData, Variable<Vector> w, VariableArray<bool> y)
        {
            Range j = y.Range.Named("numSamples");
            VariableArray<Vector> x = Variable.Observed(xData, j).Named("x");

            //******************************************************************
            // Bayes Point Machine model
            //******************************************************************

            double noise = 0.1;

            using (Variable.ForEach(j))
            {
                y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise) > 0;
            }
        }

        static void Main(string[] args)
        {
            //**************************************************************************************************************
            // Notes:
            //   (1) This script takes data from a csv file (without headers or index columns):
            //       (1.1) training data from csv
            //       (1.2) testing data from csv
            //   The data.csv file contains all the features X, and the last column the target Y.
            //***************************************************************************************************************/

            // the data.csv file contains all the features X, and last column the target Y.
            var fileName = "creditcard-train.csv";
            var lines = File.ReadAllLines(fileName);
            bool[] trainDataY = new bool[lines.Length];
            Vector[] trainVectorX = new Vector[lines.Length];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] strArray = lines[i].Split(';');
                double[] doubleArray = Array.ConvertAll(strArray, double.Parse);
                double[] doubleXArray = new double[4];
                int dims = doubleArray.Length;

                // create training x vector, augmented by 1 for bias term
                doubleXArray[0] = doubleArray[0];
                doubleXArray[1] = doubleArray[1];
                doubleXArray[2] = doubleArray[2];
                doubleXArray[3] = 1.0;
                //doubleXArray[dims - 1] = 1.0;
                trainVectorX[i] = Vector.FromArray(doubleXArray);

                if (doubleArray[dims - 1] == 1)
                {
                    trainDataY[i] = true;
                }
                else
                {
                    trainDataY[i] = false;
                }
            }

            fileName = "creditcard-test.csv";
            lines = File.ReadAllLines(fileName);
            bool[] testDataY = new bool[lines.Length];
            Vector[] testVectorX = new Vector[lines.Length];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] strArray = lines[i].Split(';');
                double[] doubleArray = Array.ConvertAll(strArray, double.Parse);
                double[] doubleXArray = new double[4];
                int dims = doubleArray.Length;

                // create training x vector, augmented by 1 for bias term
                doubleXArray[0] = doubleArray[0];
                doubleXArray[1] = doubleArray[1];
                doubleXArray[2] = doubleArray[2];
                doubleXArray[3] = 1.0;
                //doubleXArray[dims - 1] = 1.0;
                testVectorX[i] = Vector.FromArray(doubleXArray);

                if (doubleArray[dims - 1] == 1)
                {
                    testDataY[i] = true;
                }
                else
                {
                    testDataY[i] = false;
                }

            }

            //******************************************************************
            // Train the Bayes Point Machine model
            //******************************************************************

            VariableArray<bool> y = Variable.Observed(trainDataY).Named("y");

            int numDims = trainVectorX[0].Count;
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(numDims), PositiveDefiniteMatrix.Identity(numDims))).Named("w");

            BayesPointMachine(trainVectorX, w, y);

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.NumberOfIterations = 50;
            engine.ShowFactorGraph = false;

            // infer the posterior distribution over the weights
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);

            Console.WriteLine("Dist over w=\n" + wPosterior);

            //******************************************************************
            // Test the Bayes Point Machine model
            //******************************************************************

            VariableArray<bool> yTest = Variable.Array<bool>(new Range(testDataY.Length)).Named("yTest");

            BayesPointMachine(testVectorX, Variable.Random(wPosterior).Named("w"), yTest);

            // infer the posterior distributions over the target variable
            Bernoulli[] yTestPost = engine.Infer<Bernoulli[]>(yTest);

            //for (int i = 0; i < testVectorX.Length; i++)
            //{
            //    Console.WriteLine("Predicted: {0}, Actual: {1}", trainDataY[i], yTestPost[i]);
            //}

            //*********** store outputs ***********/
            var storeWeightsMeans = new StringBuilder();
            for (int i = 0; i < wPosterior.GetMean().Count(); i++)
            {
                var line = string.Format("{0}", wPosterior.GetMean()[i]);
                storeWeightsMeans.AppendLine(line);
            }

            var storeWeightsVariance = new StringBuilder();
            for (int i = 0; i < wPosterior.GetVariance().Count(); i++)
            {
                var line = string.Format("{0}", wPosterior.GetVariance()[i]);
                storeWeightsVariance.AppendLine(line);
            }

            var storeTargetPredictions = new StringBuilder();
            for (int i = 0; i < testDataY.Length; i++)
            {
                var line = string.Format("{0};{1}", yTestPost[i].GetProbTrue(), testDataY[i]);
                storeTargetPredictions.AppendLine(line);
            }

            File.WriteAllText("test-predictions.csv", storeTargetPredictions.ToString());
            File.WriteAllText("weights-means.csv", storeWeightsMeans.ToString());
            File.WriteAllText("weights-variances.csv", storeWeightsVariance.ToString());
        }
    }
    
}