require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const loadCSV = require('../load-csv');

const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
  }
);

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);

const linearRegression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

linearRegression.train();

/*
  console.log(
    'm:',
    linearRegression.weights.arraySync()[1][0],
    'b:',
    linearRegression.weights.arraySync()[0][0]
  );
*/

const r2 = linearRegression.test(testFeatures, testLabels);

console.log(linearRegression.mseHistory);

console.log(r2);

linearRegression.predict([[120, 2, 380], [135, 2.1, 420]]).print();

/*
  // console.log(linearRegression.bHistory);

  const mseHistoryReversed = linearRegression.mseHistory.reverse();

  plot({
    name: 'mse_iteration',
    x: mseHistoryReversed,
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error (MSE)',
  });

  plot({
    name: 'mse_b',
    x: linearRegression.bHistory.reverse(),
    y: mseHistoryReversed,
    xLabel: 'b',
    yLabel: 'Mean Squared Error (MSE)',
  });
*/
