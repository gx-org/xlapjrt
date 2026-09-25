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
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	pjbfloat16 "github.com/gomlx/gopjrt/dtypes/bfloat16"
	pjtypes "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
	dtype "github.com/gx-org/backend/dtypes"
	gxfmt "github.com/gx-org/gx/base/fmt"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
	pjrtgx "github.com/gx-org/xlapjrt"
)

type (
	// Graph is the PJRT compute graph.
	Graph struct {
		inputs *tuple

		plat       *pjrtplatform.Platform
		builder    *xlabuilder.XlaBuilder
		executable *pjrt.LoadedExecutable

		in     []*Node
		out    []shapes.Shape
		traced []shapes.Shape
	}

	pjrtNode interface {
		backend.Value

		xlaOp() *xlabuilder.Op

		BackendShape() shapes.Shape
	}
)

var (
	_ backend.Function = (*Graph)(nil)
)

// New returns a new graph.
func New(plat *pjrtplatform.Platform, funcName string, shapes []shapes.Shape) (*Graph, error) {
	return newGraph(plat, shapes, xlabuilder.New(funcName))
}

func newGraph(plat *pjrtplatform.Platform, shapes []shapes.Shape, builder *xlabuilder.XlaBuilder) (*Graph, error) {
	g := &Graph{
		plat:    plat,
		builder: builder,
	}
	var err error
	g.inputs, err = g.buildTupleArgument(shapes)
	if err != nil {
		return nil, err
	}
	return g, nil
}

func (g *Graph) buildTupleArgument(shapes []shapes.Shape) (*tuple, error) {
	if len(shapes) == 0 {
		return nil, nil
	}
	xlaShape := make([]xlabuilder.Shape, 0, len(shapes))
	for _, shape := range shapes {
		xlaShape = append(xlaShape, pjrtgx.ToShape(shape))
	}
	const argTuple = "argtuple"
	xlaOp, err := xlabuilder.Parameter(g.builder, argTuple, 0, xlabuilder.Shape{TupleShapes: xlaShape})
	if err != nil {
		return nil, err
	}
	return &tuple{Node: g.newNode(xlaOp).Info(argTuple)}, nil
}

func (g *Graph) tupleArgument(got shapes.Shape, name string, index int) (backend.Value, error) {
	op, err := g.inputs.element(index)
	if err != nil {
		return nil, err
	}
	want := op.BackendShape()
	if !got.Equal(want) {
		return nil, errors.Errorf("cannot create argument %d:%s: got shape %s but want %s", index, name, got, want)
	}
	return op.Info("[%s]", name), nil
}

func unpackOutput(outs []*backend.OutputNode) ([]backend.Value, []shapes.Shape) {
	nodes := make([]backend.Value, len(outs))
	shs := make([]shapes.Shape, len(outs))
	for i, out := range outs {
		nodes[i] = out.Node
		shs[i] = out.Shape
	}
	return nodes, shs
}

// Compile a node given a set of parameters and using this node as an output.
// Returns a function that will be run on a device given some inputs.
func (g *Graph) Compile(dev backend.DeviceNum, out, traced []*backend.OutputNode, params []shapes.Shape) (backend.Executable, error) {
	var outNodes, tracedNodes []backend.Value
	outNodes, g.out = unpackOutput(out)
	tracedNodes, g.traced = unpackOutput(traced)
	all := append(append([]backend.Value{}, outNodes...), tracedNodes...)
	allTuple, err := g.Tuple(all)
	if err != nil {
		return nil, err
	}
	computation, err := g.builder.Build(g.xlaHandle(allTuple))
	if err != nil {
		return nil, errors.Errorf("cannot compile graph node %T for function %s: %v", all, g.builder.Name(), err)
	}
	g.executable, err = g.plat.Client().Compile().WithComputation(computation).Done()
	if err != nil {
		return nil, errors.Errorf("cannot compile graph node %T for function %s: %v", all, g.builder.Name(), err)
	}
	return g.newNodeRunner(dev), nil
}

// OutShapes returns the expected shapes of the out nodes.
func (g *Graph) OutShapes() []shapes.Shape {
	return g.out
}

// TracedShapes returns the expected shapes of the out nodes.
func (g *Graph) TracedShapes() []shapes.Shape {
	return g.traced
}

// Platform owning the graph.
func (g *Graph) Platform() backend.Platform {
	return g.plat
}

// Graph in which nodes are created.
func (g *Graph) Graph() backend.Function {
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

	info string
	deps []backend.Value // Only used for debugging.
}

var _ pjrtNode = (*Node)(nil)

func (g *Graph) xlaHandle(input backend.Value) *xlabuilder.Op {
	return input.(pjrtNode).xlaOp()
}

func (g *Graph) xlaHandles(inputs []backend.Value) ([]*xlabuilder.Op, error) {
	hdls := make([]*xlabuilder.Op, len(inputs))
	for i, node := range inputs {
		hdls[i] = node.(pjrtNode).xlaOp()
	}
	return hdls, nil
}

func (g *Graph) newNode(op *xlabuilder.Op, deps ...backend.Value) *Node {
	return &Node{graph: g, op: op, deps: deps}
}

// Info sets some debugging information about the node.
func (n *Node) Info(format string, a ...any) *Node {
	n.info = fmt.Sprintf(format, a...)
	return n
}

// Graph to which the node belongs to.
func (n *Node) Graph() backend.Function {
	return n.graph
}

// BackendShape returns the shape inferred by a backend,
// as opposed to a shape inferred by GX.
func (n *Node) BackendShape() shapes.Shape {
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

func (n *Node) String() string {
	bld := strings.Builder{}
	bld.WriteString(n.op.Type.String())
	if len(n.info) > 0 {
		bld.WriteString(":" + n.info)
	}
	if len(n.deps) == 0 {
		return bld.String() + "\n"
	}
	bld.WriteString("{\n")
	for _, dep := range n.deps {
		bld.WriteString(gxfmt.Indent(fmt.Sprint(dep)))
	}
	bld.WriteString("}\n")
	return bld.String()
}

func newLiteral[T pjtypes.Supported](data []T, dims []int) (*xlabuilder.Literal, error) {
	if len(dims) == 0 {
		return xlabuilder.NewScalarLiteral(data[0]), nil
	}
	return xlabuilder.NewArrayLiteral(data, dims...)
}

// Constant returns a node representing a numerical constant value in the graph.
func (g *Graph) Constant(data []byte, shap shapes.Shape) (backend.Value, error) {
	var literal *xlabuilder.Literal
	var err error
	switch shap.DType {
	case dtypes.Bool:
		literal, err = newLiteral(dtype.ToSlice[bool](data), shap.Dimensions)
	case dtypes.BFloat16:
		literal, err = newLiteral(dtype.ToSlice[pjbfloat16.BFloat16](data), shap.Dimensions)
	case dtypes.Float32:
		literal, err = newLiteral(dtype.ToSlice[float32](data), shap.Dimensions)
	case dtypes.Float64:
		literal, err = newLiteral(dtype.ToSlice[float64](data), shap.Dimensions)
	case dtypes.Int32:
		literal, err = newLiteral(dtype.ToSlice[int32](data), shap.Dimensions)
	case dtypes.Int64:
		literal, err = newLiteral(dtype.ToSlice[int64](data), shap.Dimensions)
	case dtypes.Uint32:
		literal, err = newLiteral(dtype.ToSlice[uint32](data), shap.Dimensions)
	case dtypes.Uint64:
		literal, err = newLiteral(dtype.ToSlice[uint64](data), shap.Dimensions)
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

// NewAtomLiteral creates a node from a constant atom.
func (g *Graph) NewAtomLiteral(v any) (backend.Value, error) {
	var lit *xlabuilder.Literal
	var err error
	switch vT := v.(type) {
	case int:
		lit = xlabuilder.NewScalarLiteral(vT)
	case bfloat16.BFloat16:
		lit = xlabuilder.NewScalarLiteral(pjbfloat16.BFloat16(vT))
	default:
		lit, err = xlabuilder.NewScalarLiteralFromAny(vT)
	}
	if err != nil {
		return nil, err
	}
	op, err := xlabuilder.Constant(g.builder, lit)
	if err != nil {
		return nil, err
	}
	return g.newNode(op), nil
}

// NewArrayLiteral creates a node from a constant array.
func (g *Graph) NewArrayLiteral(flat any, axlengths ...int) (backend.Value, error) {
	var lit *xlabuilder.Literal
	var err error
	switch flatT := flat.(type) {
	case []bfloat16.BFloat16:
		pjFlat := make([]pjbfloat16.BFloat16, len(flatT))
		for i, v := range flatT {
			pjFlat[i] = pjbfloat16.BFloat16(v)
		}
		lit, err = xlabuilder.NewArrayLiteral(pjFlat, axlengths...)
	default:
		lit, err = xlabuilder.NewArrayLiteralFromAny(flat, axlengths...)
	}
	if err != nil {
		return nil, err
	}
	op, err := xlabuilder.Constant(g.builder, lit)
	if err != nil {
		return nil, err
	}
	return g.newNode(op), nil
}

// Argument returns a node set by a caller when calling the function.
func (g *Graph) Argument(name string, shape shapes.Shape, index int) (node backend.Value, err error) {
	if g.inputs != nil {
		return g.tupleArgument(shape, name, index)
	}
	xlaOp, err := xlabuilder.Parameter(g.builder, name, index, pjrtgx.ToShape(shape))
	if err != nil {
		return nil, err
	}
	arg := g.newNode(xlaOp).Info("%s:%d", name, index)
	g.in = append(g.in, arg)
	return arg, nil
}

// UnaryFunc returns a node executing a unary function. f must be an xlabuilder function pointer.
func (g *Graph) UnaryFunc(x backend.Value, f func(*xlabuilder.Op) (*xlabuilder.Op, error)) (backend.Value, error) {
	result, err := f(g.xlaHandle(x))
	if err != nil {
		return nil, err
	}
	return g.newNode(result, x), nil
}

// BinaryFunc returns a node executing a binary function. f must be an xlabuilder function pointer.
func (g *Graph) BinaryFunc(x backend.Value, y backend.Value, f func(x *xlabuilder.Op, y *xlabuilder.Op) (*xlabuilder.Op, error)) (backend.Value, error) {
	result, err := f(g.xlaHandle(x), g.xlaHandle(y))
	if err != nil {
		return nil, err
	}
	return g.newNode(result, x, y), nil
}

// ReduceFunc returns a node executing a basic reduction. f must be an xlabuilder function pointer.
func (g *Graph) ReduceFunc(x backend.Value, axes []int, f func(*xlabuilder.Op, ...int) (*xlabuilder.Op, error)) (backend.Value, error) {
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

// LogicalNot returns a node computing the logical not of x.
func (g *Graph) LogicalNot(x backend.Value) (backend.Value, error) {
	return g.UnaryFunc(x, xlabuilder.LogicalNot)
}

// Neg returns a node computing the negation of x.
func (g *Graph) Neg(x backend.Value) (backend.Value, error) {
	return g.UnaryFunc(x, xlabuilder.Neg)
}

// Add returns a node adding two nodes.
func (g *Graph) Add(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Add)
}

// Sub returns a node subtracting y from x.
func (g *Graph) Sub(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Sub)
}

// Mul returns a node multiplying two nodes.
func (g *Graph) Mul(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Mul)
}

// Div returns a node dividing x by y.
func (g *Graph) Div(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Div)
}

// Rem returns a node computing remainder of x divided by y.
func (g *Graph) Rem(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Rem)
}

// Equal returns a boolean node checking if x == y.
func (g *Graph) Equal(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.Equal)
}

// NotEqual returns a boolean node checking if x != y.
func (g *Graph) NotEqual(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.NotEqual)
}

// LessThan returns a boolean node checking if x < y.
func (g *Graph) LessThan(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.LessThan)
}

// LessOrEqual returns a boolean node checking if x <= y.
func (g *Graph) LessOrEqual(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.LessOrEqual)
}

// GreaterThan returns a boolean node checking if x > y.
func (g *Graph) GreaterThan(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.GreaterThan)
}

// GreaterOrEqual returns a boolean node checking if x >= y.
func (g *Graph) GreaterOrEqual(x, y backend.Value) (backend.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.GreaterOrEqual)
}

// ShiftLeft returns a node shifting x left by y.
func (g *Graph) ShiftLeft(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.ShiftLeft)
}

// ShiftRight returns a node shifting x right by y.
func (g *Graph) ShiftRight(x, y backend.Value) (backend.Value, error) {
	// We copy Go's behavior: "shift operators implement arithmetic shifts if the left operand is a
	// signed integer and logical shifts if it is an unsigned integer".
	if g.xlaHandle(x).Shape.DType.IsUnsigned() {
		return g.BinaryFunc(x, y, xlabuilder.ShiftRightLogical)
	}
	return g.BinaryFunc(x, y, xlabuilder.ShiftRightArithmetic)
}

// BitwiseAnd returns a node computing bitwise AND of x and y.
func (g *Graph) BitwiseAnd(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseAnd)
}

// BitwiseOr returns a node computing bitwise OR of x and y.
func (g *Graph) BitwiseOr(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseOr)
}

// BitwiseXor returns a node computing bitwise XOR of x and y.
func (g *Graph) BitwiseXor(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseXor)
}

// LogicalAnd returns a node computing logical AND of x and y.
func (g *Graph) LogicalAnd(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.LogicalAnd)
}

// LogicalOr returns a node computing logical OR of x and y.
func (g *Graph) LogicalOr(x, y backend.Value) (backend.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.LogicalOr)
}

// Reshape returns a reshape operator node.
func (g *Graph) Reshape(x backend.Value, axisLengths []int) (backend.Value, error) {
	xlaOp, err := xlabuilder.Reshape(g.xlaHandle(x), axisLengths...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Cast returns a cast/convert operator node.
func (g *Graph) Cast(x backend.Value, target dtypes.DType) (backend.Value, error) {
	xlaDType := pjrtgx.ToPJDType(target)
	if xlaDType == pjtypes.InvalidDType {
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
func (n *tuple) element(i int) (*Node, error) {
	xlaOp, err := xlabuilder.GetTupleElement(n.graph.xlaHandle(n.Node), i)
	if err != nil {
		return nil, err
	}
	return n.graph.newNode(xlaOp).Info("%s[%d]", n.info, i), nil
}

// Element returns a Node representing the ith element of the tuple.
func (n *tuple) Element(i int) (backend.Value, error) {
	return n.element(i)
}

func (n *tuple) Size() int {
	// Note: this relies on gopjrt's shape tracking.
	return n.Node.op.Shape.TupleSize()
}

func (n *tuple) Unpack() ([]backend.Value, error) {
	nodes := make([]backend.Value, 0, n.Size())
	for i := range n.Size() {
		node, err := n.Element(i)
		if err != nil {
			return nil, err
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

// Tuple returns a node grouping multiple nodes together.
func (g *Graph) Tuple(nodes []backend.Value) (backend.Tuple, error) {
	inputs, err := g.xlaHandles(nodes)
	if err != nil {
		return nil, err
	}
	xlaOp, err := xlabuilder.Tuple(inputs...)
	if err != nil {
		return nil, err
	}
	return &tuple{g.newNode(xlaOp, nodes...)}, nil
}

// ToXLATuple casts a generic Node to a graph.Tuple node.
func ToXLATuple(n backend.Value) backend.Tuple {
	if tpl, ok := n.(*tuple); ok {
		return tpl
	}
	return &tuple{Node: n.(*Node)}
}

// Slice returns a slice on a node.
func (g *Graph) Slice(x backend.Value, i int) (backend.Value, error) {
	shape := x.(pjrtNode).BackendShape()
	rank := len(shape.Dimensions)

	starts := make([]int, rank)
	limits := make([]int, rank)
	strides := make([]int, rank)
	for axis, axisSize := range shape.Dimensions {
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
	reshapeOp, err := xlabuilder.Reshape(sliceOp, shape.Dimensions[1:]...)
	if err != nil {
		return nil, err
	}
	return g.newNode(reshapeOp), nil
}

// BroadcastInDim broadcasts x to an output with the given shape.
func (g *Graph) BroadcastInDim(x backend.Value, shape shapes.Shape, broadcastAxes []int) (backend.Value, error) {
	xlaOp, err := xlabuilder.BroadcastInDim(g.xlaHandle(x), pjrtgx.ToShape(shape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Gather exposes the full XLA Gather operation.
func (g *Graph) Gather(x backend.Value, startIndices backend.Value, indexVectorAxis int, offsetAxes []int, collapsedSliceAxes []int, startIndexMap []int, sliceSizes []int, indicesAreSorted bool) (backend.Value, error) {
	xlaOp, err := xlabuilder.Gather(g.xlaHandle(x), g.xlaHandle(startIndices), indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Set returns a node to set a slice in an array.
func (g *Graph) Set(x, update backend.Value, position []backend.Value) (backend.Value, error) {
	xOp := g.xlaHandle(x)
	xShape := xOp.Shape
	rank := len(xShape.Dimensions)
	indexDType := g.xlaHandle(position[0]).Shape.DType
	xlaPos := make([]*xlabuilder.Op, rank)
	zeroLit, err := xlabuilder.NewScalarLiteralFromFloat64(0.0, indexDType)
	if err != nil {
		return nil, err
	}
	zeroOp, err := xlabuilder.Constant(g.builder, zeroLit)
	if err != nil {
		return nil, err
	}
	for i := 0; i < rank; i++ {
		if i < len(position) {
			xlaPos[i] = g.xlaHandle(position[i])
		} else {
			xlaPos[i] = zeroOp
		}
	}
	updateShape := make([]int, rank)
	for i := 0; i < rank; i++ {
		if i < len(position) {
			updateShape[i] = 1
		} else {
			updateShape[i] = xShape.Dimensions[i]
		}
	}
	xlaUpdate := g.xlaHandle(update)
	xlaUpdateReshaped, err := xlabuilder.Reshape(xlaUpdate, updateShape...)
	if err != nil {
		return nil, err
	}

	xlaRes, err := xlabuilder.DynamicUpdateSlice(xOp, xlaUpdateReshaped, xlaPos)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaRes), nil
}

// DotGeneral returns a generic dot product node. Batch and reduce axes are given as pairs of
// equal-length slices, left hand axes followed by right hand axes.
func (g *Graph) DotGeneral(x, y backend.Value, batchAxes, reduceAxes [2][]int) (backend.Value, error) {
	xlaOp, err := xlabuilder.DotGeneral(
		g.xlaHandle(x), reduceAxes[0], batchAxes[0],
		g.xlaHandle(y), reduceAxes[1], batchAxes[1])
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Call returns a node that invokes a subgraph with the given result node.
func (g *Graph) Call(sg *backend.Subgraph, args ...backend.Value) (backend.Value, error) {
	subcomp, err := g.xlaSubcomputation(sg)
	if err != nil {
		return nil, err
	}
	argOps, err := g.xlaHandles(args)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.Call(g.builder, subcomp.comp, argOps...)
	if err != nil {
		return nil, err
	}
	var result backend.Value = g.newNode(xlaOp, subcomp)
	if _, ok := sg.Result.Node.(backend.Tuple); ok {
		// If the result node was a tuple, the subgraph's return value will also be a tuple.
		result = ToXLATuple(result)
	}
	return result, nil
}

// Subgraph returns a Graph instance that maps to a new subgraph.
func (g *Graph) Subgraph(name string, inputs []shapes.Shape) (backend.Function, error) {
	subName := g.builder.Name() + "." + name
	builder := g.builder.CreateSubBuilder(subName)
	return newGraph(g.plat, inputs, builder)
}

type subGraph struct {
	out   backend.Value
	comp  *xlabuilder.XlaComputation
	graph *Graph
}

func (g *Graph) xlaSubcomputation(sg *backend.Subgraph) (*subGraph, error) {
	pjrtsg := sg.Graph.(*Graph)
	op := sg.Result.Node
	sub := &subGraph{graph: pjrtsg, out: op}
	var err error
	sub.comp, err = pjrtsg.builder.Build(g.xlaHandle(op))
	if err != nil {
		return nil, errors.Errorf("cannot build a subgraph: %v\nSubgraph:\n%s", err, sub.String())
	}
	return sub, nil
}

func (sub *subGraph) Graph() backend.Function {
	return sub.graph
}

func (sub *subGraph) String() string {
	bld := strings.Builder{}
	fmt.Fprintf(&bld, "SUBGRAPH(%s){\n", sub.graph.builder.Name())
	for i, arg := range sub.graph.in {
		bld.WriteString(gxfmt.Indent(fmt.Sprintf("%d->%s", i, arg)))
	}
	bld.WriteString(gxfmt.Indent(fmt.Sprint(sub.out)))
	bld.WriteString("}\n")
	return bld.String()
}

// While returns a while loop node.
func (g *Graph) While(cond, body *backend.Subgraph, state backend.Value) (backend.Value, error) {
	condSG, err := g.xlaSubcomputation(cond)
	if err != nil {
		return nil, err
	}
	bodySG, err := g.xlaSubcomputation(body)
	if err != nil {
		return nil, err
	}

	xlaOp, err := xlabuilder.While(g.xlaHandle(state), condSG.comp, bodySG.comp)
	if err != nil {
		return nil, err
	}
	var result backend.Value = g.newNode(xlaOp, condSG, bodySG)
	if _, ok := state.(backend.Tuple); ok {
		result = ToXLATuple(result)
	}
	return result, nil
}

// String representation of the graph.
func (g *Graph) String() string {
	return fmt.Sprintf("XLAGraph(%q):%p", g.builder.Name(), g.builder)
}
