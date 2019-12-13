const Network = require('./').Network
const network = new Network([2, 6, 4, 1])
const epochs = 100000

for (let i = 0; i < epochs; i++) {
  network.train({
    input: [0, 1],
    output: [1]
  }, 0.1)

  network.train({
    input: [1, 0],
    output: [1]
  },  0.1)

  network.train({
    input: [1, 1],
    output: [0]
  }, 0.1)

  network.train({
    input: [0, 0],
    output: [0]
  }, 0.1)
}

console.log(`[0,1], ${JSON.stringify(network.predict([0, 1]))}`)
console.log(`[1,0], ${JSON.stringify(network.predict([1, 0]))}`)
console.log(`[1,1], ${JSON.stringify(network.predict([1, 1]))}`)
console.log(`[0,0], ${JSON.stringify(network.predict([0, 0]))}`)