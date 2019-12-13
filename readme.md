# 实现一个全连接网络

我们已经了解了神经网络的基本原理，不如趁热打铁来实现一下，巩固巩固。

## 实现
首先我们来定义网络节点

```javascript
class Node {
  constructor(id, activation) {
    this.id = id
    this.activation = activation
    this.inputEdges = []
    this.outputEdges = []
    this.bias = Math.random()

    this.input = null
    this.output = null

    this.delta = 0
  }
}
```

然后是节点与节点之间的连接边

```javascript
class Edge {
  constructor(startNode, endNode) {
    this.start = startNode
    this.end = endNode

    this.id = `${startNode.id}-${endNode.id}`
    this.weight = Math.random()
  }
}

```

接着是激活函数以及误差函数和它们的导数

```javascript
const Activations = {
  RELU: {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  },
  SIGMOID: {
    output: (x) => 1 / (1 + Math.exp(-x)),
    der: (x) => {
      const output = Activations.SIGMOID.output(x)
      return output * (1 - output)
    }
  }
}

const ErrorFunc = {
  error: (output, target) => (output - target) ** 2 / 2,
  der: (output, target) => output - target
}
```

开始写网络构造函数，我们的网络构造参数非常简洁，类似这样 `new Network([3,2,1], Activations.SIGMOD)`，在内部用一个二维数据存储每一层以及它们的神经元，如下：

```javascript
class Network {
  constructor(shape, activation = Activations.SIGMOID) {
    this.network = []
    this.layerLen = shape.length

    for (let layerId = 0; layerId < this.layerLen; layerId++) {
      const currentLayer = []

      this.network.push(currentLayer)

      for (let i = 0; i < shape[layerId]; i++) {
        const node = new Node(`${layerId}_${i}`, activation)
        currentLayer.push(node)

        if (layerId > 0) {
          for (const prevNode of this.network[layerId - 1]) {
            const edge = new Edge(prevNode, node)
            prevNode.outputEdges.push(edge)
            node.inputEdges.push(edge)
          }
        }
      }
    }
  }
}
```
我们定义一个简单的打印一下看看结果

```javascript
console.log(new Network([3,2,4]).network[0][0].outputEdges)

[
  Edge {
    start: Node {
      id: '0_0',
      activation: [Object],
      inputEdges: [],
      outputEdges: [Circular],
      bias: 0.7207774062670897,
      input: null,
      output: null,
      delta: 0
    },
    end: Node {
      id: '1_0',
      activation: [Object],
      inputEdges: [Array],
      outputEdges: [Array],
      bias: 0.6852813481404556,
      input: null,
      output: null,
      delta: 0
    },
    id: '0_0-1_0',
    weight: 0.7299873105671102
  },
  Edge {
    start: Node {
      id: '0_0',
      activation: [Object],
      inputEdges: [],
      outputEdges: [Circular],
      bias: 0.7207774062670897,
      input: null,
      output: null,
      delta: 0
    },
    end: Node {
      id: '1_1',
      activation: [Object],
      inputEdges: [Array],
      outputEdges: [Array],
      bias: 0.1261033454333067,
      input: null,
      output: null,
      delta: 0
    },
    id: '0_0-1_1',
    weight: 0.4034807153423452
  }
]
```

可以看到第一层的第一个输入节点正确的与第二层的两个节点相连，下面我们来进行正向传播，给 Newwork 增加 `forward` 函数

```javascript
  forward(inputs) {
    // 更新输入层
    this.network[0].forEach((node, i) => {
      node.output = inputs[i]
      node.input = inputs[i]
    })

    // 计算每个节点输出
    for (let i = 1; i < this.layerLen; i++) {
      for (const node of this.network[i]) {
        let rst = node.bias
        for (const edge of node.inputEdges) {
          rst += edge.weight * edge.start.output
        }
        node.input = rst
        node.output = node.activation.output(rst)
      }
    }
  }
```

接下来是重点的反向传播，输入目标值，反向传播网络最终输出层的误差，最后一层使用误差函数偏导，隐藏层每个节点的误差是和它相连的后一层节点误差之和，最后通过激活函数偏导求一下在该节点上需要更新的分量`delta`：

```javascript
  backward(target) {
    for (let layerId = this.layerLen - 1; layerId >= 1; layerId--) {
      const currentLayer = this.network[layerId]

      currentLayer.forEach((node, i) => {
        let error = 0
        if (layerId === this.layerLen - 1) {
          error = ErrorFunc.d(node.output, target[i])
        } else {
          for (let edge of node.outputEdges) {
            error += edge.weight * edge.end.delta
          }
        }

        node.delta = error * node.activation.d(node.output)
      })
    }
  }
```

算出单节点更新分量后，就可以更新节点间权重，与该节点连接的边上的权重更新可以表示为`edge.weight += rate * node.delta * edge.start.output`，`rate`为学习率，权重与上一个节点输入有关，还需要有`edge.start.output`，对于节点上的`bias`，更新的时候直接使用 ` node.bias += rate * node.delta` 即可。


```javascript
  updateWeights(rate) {
    for (let i = 1; i < this.layerLen; i++) {
      const currentLayer = this.network[i]

      for (const node of currentLayer) {
        for (const edge of node.inputEdges) {
          edge.weight += rate * node.delta * edge.start.output
        }

        node.bias += rate * node.delta
      }
    }
  }
```

最后再定义几个便捷方法

```javascript
  train(data, learningRate) {
    this.forward(data.input)
    this.backward(data.output)
    this.updateWeights(learningRate)
    return this.getResult()
  }

  getResult(){
    return this.network[this.layerLen - 1].map((node) => {
      return node.output
    })
  }

  predict(input) {
    this.forward(input)
    return this.getResult()
  }
```

## 测试

### xor
我们用经典的 xor 问题来测试一下 

```javascript
const Network = require('./index').Network
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
```

输出，结果符合预期：

```
[0,1], [0.9965230909855941]
[1,0], [0.9956413400526261]
[1,1], [0.002851951618435603]
[0,0], [0.0029121450247287365]
```
### mnist
使用经典的手写数据，训练集 80000，测试集 8000 
```javascript
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
```
结果看还可以，测试集上正确率 84.25%

```
-learningRate: 0.1
Training accuracy: 2367/8000, 29.5875%
Training accuracy: 8541/16000, 53.38125%
Training accuracy: 14975/24000, 62.39583333333333%
Training accuracy: 21647/32000, 67.646875%
Training accuracy: 28397/40000, 70.9925%
Training accuracy: 35232/48000, 73.4%
Training accuracy: 42069/56000, 75.12321428571428%
Training accuracy: 48934/64000, 76.459375%
Training accuracy: 55823/72000, 77.53194444444445%
Training accuracy: 62778/80000, 78.4725%
Test accuracy: 674/800, 84.25%
```