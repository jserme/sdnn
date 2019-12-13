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

class Edge {
  constructor(startNode, endNode) {
    this.start = startNode
    this.end = endNode

    this.id = `${startNode.id}-${endNode.id}`
    this.weight = Math.random()
  }
}

const Activations = {
  RELU: {
    output: x => Math.max(0, x),
    d: x => x <= 0 ? 0 : 1
  },
  SIGMOID: {
    output: (x) => 1 / (1 + Math.exp(-x)),
    d: (x) => {
      const output = Activations.SIGMOID.output(x)
      return output * (1 - output)
    }
  }
}

const ErrorFunc = {
  error: (output, target) => (output - target) ** 2 / 2,
  d: (output, target) =>  target - output
}

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

  // 误差反向传播
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
}

exports.Network = Network