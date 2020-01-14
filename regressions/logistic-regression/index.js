require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const loadCSV = require('../load-csv');

const LogisticRegression = require('./logistic-regression');

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: value => {
        return value === 'TRUE' ? 1 : 0;
      },
    },
  }
);

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);

const logisticRegression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
});

logisticRegression.train();

console.log(logisticRegression.costHistory);

console.log(logisticRegression.test(testFeatures, testLabels));

plot({
  x: logisticRegression.costHistory.reverse(),
  name: 'cost_ce_iteration',
  xLabel: 'Iteration #',
  yLabel: 'Cost (Cross Entropy)',
});

logisticRegression
  .predict([
    [130, 1.75, 307],
    [88, 1.065, 97],
  ])
  .print();
