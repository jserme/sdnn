const Network = require('.').Network
const network = new Network([784, 30, 10])
const mnist = require('mnist')

const TRAINING_SIZE = 8000
const TEST_SIZE = 800

const set = mnist.set(TRAINING_SIZE, TEST_SIZE)

const trainIterations = 10
const learningRate = 0.1

let trainingCorrect = 0
let testCorrect = 0
let trainingCount = 0

function getMax(arr) {
  return arr.indexOf(Math.max(...arr))
}

console.log(`-learningRate: ${learningRate}`)

for (let i = 0; i < trainIterations; i++) {
  for (const data of set.training) {
    let r = network.train({
      input: data.input,
      output: data.output
    }, learningRate)

    trainingCount++

    if (getMax(r) === getMax(data.output)) {
      trainingCorrect++
    }
  }
  console.log(`Training accuracy: ${trainingCorrect}/${trainingCount}, ${trainingCorrect/trainingCount*100}%`)
}

for (const data of set.test) {
  let r = network.predict(data.input)
  if (getMax(r) === getMax(data.output)) {
    testCorrect++
  }
}

console.log(`Test accuracy: ${testCorrect}/${TEST_SIZE}, ${testCorrect/TEST_SIZE*100}%`)