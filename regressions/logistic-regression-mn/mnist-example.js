require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const LogisticRegression = require('./logistic-regression-mn');

const loadLR = () => {
  const trainingMnistData = mnist.training(0, 60000); // Training observations from i1 to i2-1

  const features = trainingMnistData.images.values.map(image =>
    _.flatMap(image)
  );
  const labels = trainingMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 500,
    batchSize: 500,
  });
};

const testMnistData = mnist.testing(0, 1000); // Test observations from i1 to i2-1

const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testLabels = testMnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const logisticRegression = loadLR();

logisticRegression.train();

console.log(logisticRegression.test(testFeatures, testLabels));

plot({
  x: logisticRegression.costHistory.reverse(),
  name: 'cost_ce_iteration_mnist',
  xLabel: 'Iteration #',
  yLabel: 'Cost (Cross Entropy)',
});
