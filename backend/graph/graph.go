// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package graph builds a PJRT graph.
package graph

import (
	"go/ast"
	"go/token"

	"github.com/pkg/errors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/dtype"
	"github.com/gx-org/backend/graph"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/build/ir"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
	pjrtgx "github.com/gx-org/xlapjrt"
)

type (
	// Graph is the PJRT compute graph.
	Graph struct {
		funcName   string
		plat       *pjrtplatform.Platform
		builder    *xlabuilder.XlaBuilder
		executable *pjrt.LoadedExecutable

		out    []*shape.Shape
		traced []*shape.Shape
	}

	pjrtNode interface {
		graph.Node

		xlaOp() *xlabuilder.Op

		BackendShape() *shape.Shape
	}
)

var (
	_ graph.Graph = (*Graph)(nil)
)

// New returns a new graph.
func New(plat *pjrtplatform.Platform, funcName string) graph.Graph {
	return &Graph{
		plat:     plat,
		builder:  xlabuilder.New(funcName),
		funcName: funcName,
	}
}

func unpackOutput(outs []*graph.OutputNode) ([]graph.Node, []*shape.Shape) {
	nodes := make([]graph.Node, len(outs))
	shapes := make([]*shape.Shape, len(outs))
	for i, out := range outs {
		nodes[i] = out.Node
		shapes[i] = out.Shape
	}
	return nodes, shapes
}

// Compile a node given a set of parameters and using this node as an output.
// Returns a function that will be run on a device given some inputs.
func (g *Graph) Compile(dev platform.Device, out, traced []*graph.OutputNode, params []*shape.Shape) (graph.Runner, error) {
	var outNodes, tracedNodes []graph.Node
	outNodes, g.out = unpackOutput(out)
	tracedNodes, g.traced = unpackOutput(traced)
	all := append(append([]graph.Node{}, outNodes...), tracedNodes...)
	allTuple, err := g.NewTuple(all)
	if err != nil {
		return nil, err
	}
	computation, err := g.builder.Build(g.xlaHandle(allTuple))
	if err != nil {
		return nil, errors.Errorf("cannot compile graph node %T for function %s: %v", all, g.funcName, err)
	}
	g.executable, err = g.plat.Client().Compile().WithComputation(computation).Done()
	if err != nil {
		return nil, errors.Errorf("cannot compile graph node %T for function %s: %v", all, g.funcName, err)
	}
	return g.newNodeRunner(dev.(*pjrtplatform.Device)), nil
}

// OutShapes returns the expected shapes of the out nodes.
func (g *Graph) OutShapes() []*shape.Shape {
	return g.out
}

// TracedShapes returns the expected shapes of the out nodes.
func (g *Graph) TracedShapes() []*shape.Shape {
	return g.traced
}

// Platform owning the graph.
func (g *Graph) Platform() platform.Platform {
	return g.plat
}

// Graph in which nodes are created.
func (g *Graph) Graph() graph.Graph {
	return g
}

// Core returns the builder to build core operations.
func (g *Graph) Core() graph.CoreBuilder {
	return g
}

// Executable returns the PJRT executable.
func (g *Graph) Executable() *pjrt.LoadedExecutable {
	return g.executable
}

// Node in a XLA graph.
type Node struct {
	graph *Graph
	op    *xlabuilder.Op
}

var _ pjrtNode = (*Node)(nil)

func (g *Graph) xlaHandle(input graph.Node) *xlabuilder.Op {
	return input.(pjrtNode).xlaOp()
}

func (g *Graph) xlaHandles(inputs []graph.Node) ([]*xlabuilder.Op, error) {
	hdls := make([]*xlabuilder.Op, len(inputs))
	for i, node := range inputs {
		hdls[i] = node.(pjrtNode).xlaOp()
	}
	return hdls, nil
}

func (g *Graph) newNode(op *xlabuilder.Op) *Node {
	return &Node{graph: g, op: op}
}

// Graph to which the node belongs to.
func (n *Node) Graph() graph.Graph {
	return n.graph
}

// BackendShape returns the shape inferred by a backend,
// as opposed to a shape inferred by GX.
func (n *Node) BackendShape() *shape.Shape {
	return pjrtgx.ToGXShape(n.op.Shape)
}

// PJRTDims returns the dimension of the node computed by PJRT.
// TODO(degris): remove once the interpreter can compute the axis lengths.
//
// Deprecated: temporary function used as a workaround.
func (n *Node) PJRTDims() []int {
	return n.xlaOp().Shape.Dimensions
}

func (n *Node) xlaOp() *xlabuilder.Op {
	return n.op
}

func newLiteral[T dtypes.Supported](data []T, dims []int) (*xlabuilder.Literal, error) {
	if len(dims) == 0 {
		return xlabuilder.NewScalarLiteral(data[0]), nil
	}
	return xlabuilder.NewArrayLiteral(data, dims...)
}

// NewConstant returns a node representing a numerical constant value in the graph.
func (g *Graph) NewConstant(buffer platform.HostBuffer) (graph.Node, error) {
	data := buffer.Acquire()
	defer buffer.Release()
	shap := buffer.Shape()
	var literal *xlabuilder.Literal
	var err error
	switch shap.DType {
	case dtype.Bool:
		literal, err = newLiteral(dtype.ToSlice[bool](data), shap.AxisLengths)
	case dtype.Float32:
		literal, err = newLiteral(dtype.ToSlice[float32](data), shap.AxisLengths)
	case dtype.Float64:
		literal, err = newLiteral(dtype.ToSlice[float64](data), shap.AxisLengths)
	case dtype.Int32:
		literal, err = newLiteral(dtype.ToSlice[int32](data), shap.AxisLengths)
	case dtype.Int64:
		literal, err = newLiteral(dtype.ToSlice[int64](data), shap.AxisLengths)
	case dtype.Uint32:
		literal, err = newLiteral(dtype.ToSlice[uint32](data), shap.AxisLengths)
	case dtype.Uint64:
		literal, err = newLiteral(dtype.ToSlice[uint64](data), shap.AxisLengths)
	default:
		err = errors.Errorf("cannot create a PJRT literal: data type %v not supported", shap.DType)
	}
	if err != nil {
		return nil, err
	}
	op, err := xlabuilder.Constant(g.builder, literal)
	if err != nil {
		return nil, err
	}
	return g.newNode(op), nil
}

// NewArgument returns a node set by a caller when calling the function.
func (g *Graph) NewArgument(name string, shape *shape.Shape, index int) (graph.Node, error) {
	xlaOp, err := xlabuilder.Parameter(g.builder, name, index, pjrtgx.ToShape(shape))
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewTupleArgument returns a node representing a tuple parameter of a function.
func (g *Graph) NewTupleArgument(name string, index int, shapes []*shape.Shape) (graph.Tuple, error) {
	xlaShape := make([]xlabuilder.Shape, 0, len(shapes))
	for _, shape := range shapes {
		xlaShape = append(xlaShape, pjrtgx.ToShape(shape))
	}
	xlaOp, err := xlabuilder.Parameter(g.builder, name, index, xlabuilder.Shape{TupleShapes: xlaShape})
	if err != nil {
		return nil, err
	}
	return &tuple{Node: g.newNode(xlaOp)}, nil
}

// NewUnaryFunc returns a node executing a unary function. f must be an xlabuilder function pointer.
func (g *Graph) NewUnaryFunc(x graph.Node, f func(*xlabuilder.Op) (*xlabuilder.Op, error)) (graph.Node, error) {
	result, err := f(g.xlaHandle(x))
	if err != nil {
		return nil, err
	}
	return g.newNode(result), nil
}

// NewBinaryFunc returns a node executing a binary function. f must be an xlabuilder function pointer.
func (g *Graph) NewBinaryFunc(x graph.Node, y graph.Node, f func(x *xlabuilder.Op, y *xlabuilder.Op) (*xlabuilder.Op, error)) (graph.Node, error) {
	result, err := f(g.xlaHandle(x), g.xlaHandle(y))
	if err != nil {
		return nil, err
	}
	return g.newNode(result), nil
}

// NewReduceFunc returns a node executing a basic reduction. f must be an xlabuilder function pointer.
func (g *Graph) NewReduceFunc(x graph.Node, axes []int, f func(*xlabuilder.Op, ...int) (*xlabuilder.Op, error)) (graph.Node, error) {
	// Note the change from XLA's behavior: if no reduction axes are specified, treat this as a no-op.
	if len(axes) == 0 {
		return x, nil
	}
	xlaOp, err := f(g.xlaHandle(x), axes...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewUnary returns a node applying a unary operator to a node.
func (g *Graph) NewUnary(op *ast.UnaryExpr, x graph.Node) (graph.Node, error) {
	var xlaOp *xlabuilder.Op
	var err error
	switch op.Op {
	case token.SUB:
		xlaOp, err = xlabuilder.Neg(g.xlaHandle(x))
	case token.NOT:
		xlaOp, err = xlabuilder.LogicalNot(g.xlaHandle(x))
	default:
		return nil, errors.Errorf("operator %s not supported", op.Op)
	}
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewBinary returns a node applying a binary operator between two nodes.
func (g *Graph) NewBinary(op *ast.BinaryExpr, x, y graph.Node) (graph.Node, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	var xlaOp *xlabuilder.Op
	var err error
	switch op.Op {
	case token.ADD:
		xlaOp, err = xlabuilder.Add(g.xlaHandle(x), g.xlaHandle(y))
	case token.SUB:
		xlaOp, err = xlabuilder.Sub(g.xlaHandle(x), g.xlaHandle(y))
	case token.MUL:
		xlaOp, err = xlabuilder.Mul(g.xlaHandle(x), g.xlaHandle(y))
	case token.QUO:
		xlaOp, err = xlabuilder.Div(g.xlaHandle(x), g.xlaHandle(y))
	case token.EQL:
		xlaOp, err = xlabuilder.Equal(g.xlaHandle(x), g.xlaHandle(y))
	case token.GTR:
		xlaOp, err = xlabuilder.GreaterThan(g.xlaHandle(x), g.xlaHandle(y))
	case token.LSS:
		xlaOp, err = xlabuilder.LessThan(g.xlaHandle(x), g.xlaHandle(y))
	case token.NEQ:
		xlaOp, err = xlabuilder.NotEqual(g.xlaHandle(x), g.xlaHandle(y))
	case token.LEQ:
		xlaOp, err = xlabuilder.LessOrEqual(g.xlaHandle(x), g.xlaHandle(y))
	case token.GEQ:
		xlaOp, err = xlabuilder.GreaterOrEqual(g.xlaHandle(x), g.xlaHandle(y))
	case token.REM:
		xlaOp, err = xlabuilder.Rem(g.xlaHandle(x), g.xlaHandle(y))
	default:
		return nil, errors.Errorf("operator %s not supported", op.Op)
	}
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewReshape returns a reshape operator node.
func (g *Graph) NewReshape(x graph.Node, axisLengths []int) (graph.Node, error) {
	xlaOp, err := xlabuilder.Reshape(g.xlaHandle(x), axisLengths...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewCast returns a cast/convert operator node.
func (g *Graph) NewCast(x graph.Node, target dtype.DataType) (graph.Node, error) {
	xlaDType := pjrtgx.ToDType(target)
	if xlaDType == dtypes.InvalidDType {
		return nil, errors.Errorf("cannot convert %s to a XLA data type", target.String())
	}
	xlaOp, err := xlabuilder.ConvertDType(g.xlaHandle(x), xlaDType)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

type tuple struct {
	*Node
}

// Element returns a Node representing the ith element of the tuple.
func (n *tuple) Element(i int) (graph.Node, error) {
	xlaOp, err := xlabuilder.GetTupleElement(n.graph.xlaHandle(n.Node), i)
	if err != nil {
		return nil, err
	}
	return n.graph.newNode(xlaOp), nil
}

func (n *tuple) Size() int {
	// Note: this relies on gopjrt's shape tracking.
	return n.Node.op.Shape.TupleSize()
}

func (n *tuple) Unpack() ([]graph.Node, error) {
	nodes := make([]graph.Node, 0, n.Size())
	for i := range n.Size() {
		node, err := n.Element(i)
		if err != nil {
			return nil, err
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

// NewTuple returns a node grouping multiple nodes together.
func (g *Graph) NewTuple(nodes []graph.Node) (graph.Tuple, error) {
	inputs, err := g.xlaHandles(nodes)
	if err != nil {
		return nil, err
	}
	xlaOp, err := xlabuilder.Tuple(inputs...)
	if err != nil {
		return nil, err
	}
	return &tuple{g.newNode(xlaOp)}, nil
}

// ToXLATuple casts a generic Node to a graph.Tuple node.
func ToXLATuple(n graph.Node) graph.Tuple {
	if tpl, ok := n.(*tuple); ok {
		return tpl
	}
	return &tuple{Node: n.(*Node)}
}

// NewConcat concatenates multiple arrays into a single array.
func (g *Graph) NewConcat(axis int, nodes []graph.Node) (graph.Node, error) {
	inputs, err := g.xlaHandles(nodes)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.Concatenate(axis, inputs...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewSlice returns a slice on a node.
func (g *Graph) NewSlice(x graph.Node, i int) (graph.Node, error) {
	shape := x.(pjrtNode).BackendShape()
	rank := len(shape.AxisLengths)

	starts := make([]int, rank)
	limits := make([]int, rank)
	strides := make([]int, rank)
	for axis, axisSize := range shape.AxisLengths {
		starts[axis] = 0
		limits[axis] = axisSize
		strides[axis] = 1
	}

	starts[0] = i
	limits[0] = i + 1

	sliceOp, err := xlabuilder.Slice(g.xlaHandle(x), starts, limits, strides)
	if err != nil {
		return nil, err
	}
	// Slice doesn't reduce rank, so insert an additional Reshape to handle it.
	reshapeOp, err := xlabuilder.Reshape(sliceOp, shape.AxisLengths[1:]...)
	if err != nil {
		return nil, err
	}
	return g.newNode(reshapeOp), nil
}

// NewTranspose transposes the axes of x.
func (g *Graph) NewTranspose(x graph.Node, permutation []int) (graph.Node, error) {
	xlaOp, err := xlabuilder.Transpose(g.xlaHandle(x), permutation...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewIota creates a constant of the given shape with increasing numbers (starting from 0) on the given axis.
func (g *Graph) NewIota(shape *shape.Shape, iotaAxis int) (graph.Node, error) {
	xlaOp, err := xlabuilder.Iota(g.builder, pjrtgx.ToShape(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewArgMinMax returns a new argmin/argmax node.
func (g *Graph) NewArgMinMax(x graph.Node, axis int, outputKind ir.Kind, isMin bool) (graph.Node, error) {
	xlaOp, err := xlabuilder.ArgMinMax(g.xlaHandle(x), axis, pjrtgx.ToDType(outputKind.DType()), isMin)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewBroadcastInDim broadcasts x to an output with the given shape.
func (g *Graph) NewBroadcastInDim(x graph.Node, shape *shape.Shape, broadcastAxes []int) (graph.Node, error) {
	xlaOp, err := xlabuilder.BroadcastInDim(g.xlaHandle(x), pjrtgx.ToShape(shape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewGather exposes the full XLA Gather operation.
func (g *Graph) NewGather(x graph.Node, startIndices graph.Node, indexVectorAxis int, offsetAxes []int, collapsedSliceAxes []int, startIndexMap []int, sliceSizes []int, indicesAreSorted bool) (graph.Node, error) {
	xlaOp, err := xlabuilder.Gather(g.xlaHandle(x), g.xlaHandle(startIndices), indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewSplit implements the split operation in terms of slice, with static indices.
func (g *Graph) NewSplit(x graph.Node, axis int, numSplits int) (graph.Node, error) {
	shap := x.(pjrtNode).BackendShape()
	rank := len(shap.AxisLengths)

	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("axis %d is out of bounds for rank %d", axis, rank)
	}
	if shap.AxisLengths[axis]%numSplits != 0 {
		return nil, errors.Errorf("axis %d has size %d which is not divisible by %d numSplits", axis, shap.AxisLengths[axis], numSplits)
	}
	stride := shap.AxisLengths[axis] / numSplits
	slicedNodes := make([]graph.Node, numSplits)
	for i := range numSplits {
		starts := make([]int, rank)
		limits := make([]int, rank)
		strides := make([]int, rank)
		for axis, axisSize := range shap.AxisLengths {
			limits[axis] = axisSize
			strides[axis] = 1
		}

		starts[axis] = i * stride
		limits[axis] = i*stride + stride
		xlaOp, err := xlabuilder.Slice(g.xlaHandle(x), starts, limits, strides)
		if err != nil {
			return nil, err
		}

		slicedNodes[i] = g.newNode(xlaOp)
	}

	outputDims := append([]int{1}, shap.AxisLengths...)
	outputDims[axis+1] = stride

	reshapedNodes := make([]graph.Node, numSplits)
	for i := range slicedNodes {
		reshapedNode, err := g.NewReshape(slicedNodes[i], outputDims)
		if err != nil {
			return nil, err
		}
		reshapedNodes[i] = reshapedNode
	}

	return g.NewConcat(0, reshapedNodes)
}

// NewSet returns a node to set a slice in an array.
func (g *Graph) NewSet(x, updates, position graph.Node) (graph.Node, error) {
	var indexVectorDim int

	updatesShape := updates.(pjrtNode).BackendShape()
	updateWindowDims := make([]int, len(updatesShape.AxisLengths))
	for i := range len(updateWindowDims) {
		updateWindowDims[i] = i
	}
	positionShape := position.(pjrtNode).BackendShape()
	insertedWindowDims := make([]int, positionShape.AxisLengths[0])
	scatterDimsToOperandDims := make([]int, positionShape.AxisLengths[0])
	for i := range len(scatterDimsToOperandDims) {
		insertedWindowDims[i] = i
		scatterDimsToOperandDims[i] = i
	}

	const indicesAreSorted, uniqueIndices = true, true
	xlaOp, err := xlabuilder.ScatterAdd(g.xlaHandle(x), g.xlaHandle(position), g.xlaHandle(updates),
		indexVectorDim, updateWindowDims, insertedWindowDims, scatterDimsToOperandDims,
		indicesAreSorted, uniqueIndices)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewDotGeneral returns a generic dot product node. Batch and reduce axes are given as pairs of
// equal-length slices, left hand axes followed by right hand axes.
func (g *Graph) NewDotGeneral(x, y graph.Node, batchAxes, reduceAxes [2][]int) (graph.Node, error) {
	xlaOp, err := xlabuilder.DotGeneral(
		g.xlaHandle(x), reduceAxes[0], batchAxes[0],
		g.xlaHandle(y), reduceAxes[1], batchAxes[1])
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// NewCall returns a node that invokes a subgraph with the given result node.
func (g *Graph) NewCall(sg graph.Subgraph, args ...graph.Node) (graph.Node, error) {
	subcomp, err := g.xlaSubcomputation(sg)
	if err != nil {
		return nil, err
	}
	argOps, err := g.xlaHandles(args)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.Call(g.builder, subcomp, argOps...)
	if err != nil {
		return nil, err
	}
	var result graph.Node = g.newNode(xlaOp)
	if _, ok := sg.Result.Node.(graph.Tuple); ok {
		// If the result node was a tuple, the subgraph's return value will also be a tuple.
		result = ToXLATuple(result)
	}
	return result, nil
}

// NewSubgraph returns a Graph instance that maps to a new subgraph.
func (g *Graph) NewSubgraph(name string) (graph.Graph, error) {
	return &Graph{
		plat:    g.plat,
		builder: g.builder.CreateSubBuilder(name),
	}, nil
}

// NewRngBitGenerator takes RNG state and generates the given shape filled with random values, and
// returns the new state plus generated values.
func (g *Graph) NewRngBitGenerator(state graph.Node, shape *shape.Shape) (graph.Node, graph.Node, error) {
	newState, values, err := xlabuilder.RngBitGenerator(g.xlaHandle(state), pjrtgx.ToShape(shape))
	if err != nil {
		return nil, nil, err
	}
	return g.newNode(newState), g.newNode(values), nil
}

func (g *Graph) xlaSubcomputation(sg graph.Subgraph) (*xlabuilder.XlaComputation, error) {
	pjrtsg := sg.Graph.(*Graph)
	return pjrtsg.builder.Build(pjrtsg.xlaHandle(sg.Result.Node))
}

// NewWhile returns a while loop node.
func (g *Graph) NewWhile(cond, body graph.Subgraph, state graph.Node) (graph.Node, error) {
	condSG, err := g.xlaSubcomputation(cond)
	if err != nil {
		return nil, err
	}
	bodySG, err := g.xlaSubcomputation(body)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.While(g.xlaHandle(state), condSG, bodySG)
	if err != nil {
		return nil, err
	}
	var result graph.Node = g.newNode(xlaOp)
	if _, ok := state.(graph.Tuple); ok {
		result = ToXLATuple(result)
	}
	return result, nil
}
