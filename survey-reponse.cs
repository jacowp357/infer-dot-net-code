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
            int[][] sitesIndicesForEachCustomer = new int[2][] {
                new int [] { 0, 1 },
                new int [] { 0, 2 }
            };

            int[] numsitesForEachCustomer = new int[2];
            for (int n = 0; n < 2; n++)
            {
                numsitesForEachCustomer[n] = sitesIndicesForEachCustomer[n].Length;
            }

            bool[][] isDetractorAnswers = new bool[1][]
                {
                new bool [] { true, false } };

            int numDays = 1;
            int numsites = 3;
            int numCustomers = 2;

            //Range Customerssites = new Range(numsites);
            Range Days = new Range(numDays);
            Range sites = new Range(numsites);
            Range Customers = new Range(numCustomers);

            VariableArray<double> probabilityOfsiteTrue = Variable.Array<double>(sites).Named("probabilityOfsiteTrue");
            probabilityOfsiteTrue[sites] = 0.5;

            VariableArray<int> numberOfsitesForEachCustomer = Variable.Array<int>(Customers).Named("numsitesForCustomers").Attrib(new DoNotInfer());
            Range Customerssites = new Range(numberOfsitesForEachCustomer[Customers]).Named("CustomersXsites");

            VariableArray<VariableArray<int>, int[][]> sitesTouched = Variable.Array(Variable.Array<int>(Customerssites), Customers).Named("sitesTouched").Attrib(new DoNotInfer());

            VariableArray<VariableArray<bool>, bool[][]> site = Variable.Array(Variable.Array<bool>(sites), Days).Named("sites");
            site[Days][sites] = Variable.Bernoulli(probabilityOfsiteTrue[sites]).ForEach(Days);

            VariableArray<VariableArray<bool>, bool[][]> isDetractor = Variable.Array(Variable.Array<bool>(Customers), Days).Named("isDetractor");

            VariableArray<VariableArray<bool>, bool[][]> hadBadSiteInt = Variable.Array(Variable.Array<bool>(Customers), Days).Named("hadBadSiteInt");

            using (Variable.ForEach(Days))
            {
                using (Variable.ForEach(Customers))
                {
                    var relevantsites = Variable.Subarray(site[Days], sitesTouched[Customers]).Named("relevantsites");

                    //create the AllTrue factor
                    //Hassites[Days][Customers] = Variable.AllTrue(relevantsites).Named("AllTrue");

                    //create the AnyTrue factor
                    var notrelevantsites = Variable.Array<bool>(Customerssites);
                    notrelevantsites[Customerssites] = !relevantsites[Customerssites];
                    hadBadSiteInt[Days][Customers] = !Variable.AllTrue(notrelevantsites).Named("AnyTrue");

                    // add noise factor for has sites
                    using (Variable.If(hadBadSiteInt[Days][Customers]))
                    {
                        isDetractor[Days][Customers].SetTo(Variable.Bernoulli(0.8));
                    }
                    using (Variable.IfNot(hadBadSiteInt[Days][Customers]))
                    {
                        isDetractor[Days][Customers].SetTo(Variable.Bernoulli(0.2));
                    }
                }
            }

            isDetractor.ObservedValue = isDetractorAnswers;

            numberOfsitesForEachCustomer.ObservedValue = numsitesForEachCustomer;
            sitesTouched.ObservedValue = sitesIndicesForEachCustomer;

            var engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();

            Bernoulli[][] sitesPosteriors = engine.Infer<Bernoulli[][]>(site);

            for (int p = 0; p < numDays; p++)
            {
                Console.WriteLine("Day {0} sites performance distributions: ", p + 1);
                for (int s = 0; s < numsites; s++)
                {
                    Console.WriteLine(sitesPosteriors[p][s]);
                }
                Console.WriteLine("------------------");
            }
        }
    }
}
