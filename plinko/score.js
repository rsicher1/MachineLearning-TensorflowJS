const outputs = [];

const onScoreUpdate = (dropPosition, bounciness, size, bucketLabel) => {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
};

const runAnalysis = () => {
  // varying features
  const testSetSize = 100;
  const k = 10;

  _.range(0, 3).forEach(feature => {
    const outputsFeature = _.map(outputs, row => [row[feature], _.last(row)]);
    const normalizedOutputs = minMax(outputsFeature, 1);
    const [testSet, trainingSet] = splitDataset(normalizedOutputs, testSetSize);

    if (feature === 0) {
      console.log(trainingSet);
    }

    const percCorrect = _.chain(testSet)
      .filter(
        testItem =>
          knn(trainingSet, _.initial(testItem), k) === _.last(testItem)
      )
      .size()
      .divide(testSetSize)
      .value();

    console.log(
      'Feature:',
      feature,
      'Percentage correct:',
      percCorrect * 100.0
    );
  });

  /* // varying k
  const testSetSize = 100;

  const normalizedOutputs = minMax(outputs, 3);
  const [testSet, trainingSet] = splitDataset(normalizedOutputs, testSetSize);

  console.log(trainingSet);

  _.range(0, 21).forEach(k => {
    const percCorrect = _.chain(testSet)
      .filter(
        testItem =>
          knn(trainingSet, _.initial(testItem), k) === _.last(testItem)
      )
      .size()
      .divide(testSetSize)
      .value();

    console.log('k:', k, 'Percentage correct:', percCorrect * 100.0);
  });
  */
};

const distance = (pointA, pointB) => {
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
};

const splitDataset = (data, testCount) => {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
};

const knn = (data, point, k) => {
  return _.chain(data)
    .map(row => [distance(_.initial(row), point), _.last(row)])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
};

const minMax = (data, featureCount) => {
  const clonedData = _.cloneDeep(data);
  for (let i = 0; i < featureCount; i++) {
    const column = _.map(clonedData, row => row[i]);
    const min = _.min(column);
    const max = _.max(column);

    for (row of clonedData) {
      row[i] = (row[i] - min) / (max - min);
    }
  }
  return clonedData;
};
