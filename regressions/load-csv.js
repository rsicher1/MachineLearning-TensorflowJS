const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
  const headers = _.first(data);

  const indexes = _.map(columnNames, column => headers.indexOf(column));
  const extracted = _.map(data, row => _.pullAt(row, indexes));

  return extracted;
}

module.exports = function loadCSV(
  filename,
  {
    dataColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = false,
  }
) {
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });
  data = _.map(data.split('\n'), d => d.split(','));
  data = _.dropRightWhile(data, val => _.isEqual(val, ['']));
  let headers = _.first(data);
  headers = _.map(headers, element => element.replace(/"/g, ''));

  data = _.map(data, (row, index) => {
    if (index === 0) {
      return _.map(row, (element, index) => {
        return element.replace(/"/g, '');
      });
    }
    return _.map(row, (element, index) => {
      element = element.replace(/"/g, '');
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](element);
        return _.isNaN(converted) ? element : converted;
      }

      const result = parseFloat(element);
      return _.isNaN(result) ? element : result;
    });
  });

  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);

  data.shift();
  labels.shift();

  if (shuffle) {
    data = shuffleSeed.shuffle(data, 'phrase');
    labels = shuffleSeed.shuffle(labels, 'phrase');
  }

  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize),
    };
  } else {
    return { features: data, labels };
  }
};
