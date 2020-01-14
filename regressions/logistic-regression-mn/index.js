require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const _ = require('lodash');

const loadCSV = require('../load-csv');

const LogisticRegression = require('./logistic-regression-mn');

let {
  features: featuresHistorical,
  labels: labelsHistorical,
  testFeatures: testFeaturesHistorical,
  testLabels: testLabelsHistorical,
} = loadCSV(
  '/Users/rosssicherman/Documents/Job/Assessments/Columbia/GS_DBExport.csv',
  {
    // 'isfirstterm', 'issecondterm', 'isthirdterm', 'app_to_term_days', 'mpvd_count'
    dataColumns: [
      'isfirstterm',
      'issecondterm',
      'isthirdterm',
      'app_to_term_days',
      'age',
      'gpa',
    ],
    labelColumns: ['pool'],
    shuffle: true,
    splitTest: 1000,
    converters: {
      pool: value => {
        if (value === 'dual') {
          return [1, 0, 0];
        } else if (value === 'postbac') {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      },
    },
  }
);

let {
  features: featuresCurrent,
  labels: labelsCurrent,
  testFeatures: testFeaturesCurrent,
  testLabels: testLabelsCurrent,
} = loadCSV(
  '/Users/rosssicherman/Documents/Job/Assessments/Columbia/GS_DBExport_20193.csv',
  {
    dataColumns: [
      'isfirstterm',
      'issecondterm',
      'isthirdterm',
      'app_to_term_days',
      'age',
      'gpa',
    ],
    labelColumns: ['pool'],
    shuffle: true,
    splitTest: 1000,
    converters: {
      pool: value => {
        if (value === 'dual') {
          return [1, 0, 0];
        } else if (value === 'postbac') {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      },
    },
  }
);

/*
  console.log(features);
  console.log(_.flatMap(labels));
  console.log(testFeatures);
  console.log(testLabels);

  console.log(featuresHistorical);
  console.log(featuresCurrent);

  console.log(testLabelsHistorical);
  console.log(testLabelsCurrent);
*/

const logisticRegression = new LogisticRegression(
  featuresHistorical,
  _.flatMap(labelsHistorical),
  {
    learningRate: 1,
    iterations: 200,
    batchSize: 100,
  }
);

logisticRegression.train();

console.log(
  logisticRegression.test(
    testFeaturesHistorical,
    _.flatMap(testLabelsHistorical)
  )
);

console.log(
  logisticRegression.test(testFeaturesCurrent, _.flatMap(testLabelsCurrent))
);

plot({
  x: logisticRegression.costHistory.reverse(),
  name: 'cost_ce_iteration',
  xLabel: 'Iteration #',
  yLabel: 'Cost (Cross Entropy)',
});

/*
const predictions = logisticRegression.predictions.arraySync();
const actuals = logisticRegression.testLabels.arraySync();

predObj = predictions.reduce(
  (obj, pred, i) => {
    if (pred === actuals[i]) {
      obj[pred] = obj[pred] + 1;
    }
    return obj;
  },
  { 0: 0, 1: 0, 2: 0 }
);

actObj = actuals.reduce(
  (obj, act) => {
    obj[act] = obj[act] + 1;
    return obj;
  },
  { 0: 0, 1: 0, 2: 0 }
);

console.log(predObj);
console.log(actObj);
*/

/*
 1000
 age, gpa
 81.5%, 86.1%
*/

/*
 1000
 app_to_term_days, age, gpa
 82.2%, 86%
*/

/*
 1000
 isfirstterm, issecondterm, isthirdterm, age, gpa
 82.6%, 86.5%
*/

/*
 1000
 isfirstterm, issecondterm, isthirdterm, app_to_term_days, age, gpa
 82.6%, 86.9%
*/

/*
 1000
 isfirstterm, issecondterm, isthirdterm, mpvd_count, age, gpa
 82.5%, 86.4%
*/

/*
 1000
 mpvd_count, age, gpa
 81.5%, 85.9%
*/
