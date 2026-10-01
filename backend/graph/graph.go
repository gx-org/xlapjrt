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
	"google3/third_party/golang/github_com/gomlx/compute/v/v0/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	pjbfloat16 "github.com/gomlx/gopjrt/dtypes/bfloat16"
	pjtypes "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
	gxfmt "github.com/gx-org/gx/base/fmt"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
	pjrtgx "github.com/gx-org/xlapjrt"
)

type (
	// Graph is the PJRT compute graph.
	Graph struct {
		inputs *tuple

		plat       *pjrtplatform.Platform
		bld        backend.Builder
		parent     backend.Function
		builder    *xlabuilder.XlaBuilder
		executable *pjrt.LoadedExecutable

		in       []*Node
		out      []shapes.Shape
		traced   []shapes.Shape
		returned bool
		outputs  []compute.Value
	}

	pjrtNode interface {
		xlaOp() *xlabuilder.Op

		BackendShape() shapes.Shape
	}
)

var (
	_ backend.Function = (*Graph)(nil)
)

// New returns a new graph.
func New(plat *pjrtplatform.Platform, bld backend.Builder, funcName string, shapes []shapes.Shape) (*Graph, error) {
	return newGraph(plat, bld, nil, shapes, xlabuilder.New(funcName))
}

func newGraph(plat *pjrtplatform.Platform, bld backend.Builder, parent backend.Function, shapes []shapes.Shape, builder *xlabuilder.XlaBuilder) (*Graph, error) {
	g := &Graph{
		plat:    plat,
		bld:     bld,
		parent:  parent,
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

func (g *Graph) tupleArgument(got shapes.Shape, name string, index int) (*Node, error) {
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

func unpackOutput(outs []*backend.OutputNode) ([]compute.Value, []shapes.Shape) {
	nodes := make([]compute.Value, len(outs))
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
	var outNodes, tracedNodes []compute.Value
	outNodes, g.out = unpackOutput(out)
	tracedNodes, g.traced = unpackOutput(traced)
	all := append(append([]compute.Value{}, outNodes...), tracedNodes...)
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

// Name of the function.
func (g *Graph) Name() string {
	if g.parent != nil {
		return ""
	}
	return g.builder.Name()
}

// Builder returns the builder of which this function is part of.
func (g *Graph) Builder() backend.Builder {
	return g.bld
}

// Parent returns the parent function of the current function.
func (g *Graph) Parent() backend.Function {
	return g.parent
}

// Closure returns a new local function within this function.
func (g *Graph) Closure() (backend.Function, error) {
	builder := g.builder.CreateSubBuilder(g.builder.Name() + ".closure")
	return newGraph(g.plat, g.bld, g, nil, builder)
}

// Return marks the outputs of this function.
func (g *Graph) Return(outputs []compute.Value, shardings []*compute.ShardingSpec) error {
	if g.returned {
		return errors.Errorf("Return() already called for function %q", g.Name())
	}
	g.outputs = outputs
	g.returned = true
	return nil
}

// Shape returns the shape of the given Value.
func (g *Graph) Shape(v compute.Value) (shapes.Shape, error) {
	n, ok := v.(pjrtNode)
	if !ok {
		return shapes.Invalid(), errors.Errorf("invalid value type %T", v)
	}
	return n.BackendShape(), nil
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
	deps []compute.Value // Only used for debugging.
}

var _ pjrtNode = (*Node)(nil)

func (g *Graph) xlaHandle(input compute.Value) *xlabuilder.Op {
	return input.(pjrtNode).xlaOp()
}

func (g *Graph) xlaHandles(inputs []compute.Value) ([]*xlabuilder.Op, error) {
	hdls := make([]*xlabuilder.Op, len(inputs))
	for i, node := range inputs {
		hdls[i] = node.(pjrtNode).xlaOp()
	}
	return hdls, nil
}

func (g *Graph) newNode(op *xlabuilder.Op, deps ...compute.Value) *Node {
	return &Node{graph: g, op: op, deps: deps}
}

// Info sets some debugging information about the node.
func (n *Node) Info(format string, a ...any) *Node {
	n.info = fmt.Sprintf(format, a...)
	return n
}

// BackendShape returns the shape inferred by a backend,
// as opposed to a shape inferred by GX.
func (n *Node) BackendShape() shapes.Shape {
	return pjrtgx.ToGXShape(n.op.Shape)
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

// Constant creates a constant in the function with the given flat values
// and the shape defined by the dimensions.
func (g *Graph) Constant(flat any, dims ...int) (compute.Value, error) {
	var lit *xlabuilder.Literal
	var err error
	switch flatT := flat.(type) {
	case []bfloat16.BFloat16:
		pjFlat := make([]pjbfloat16.BFloat16, len(flatT))
		for i, v := range flatT {
			pjFlat[i] = pjbfloat16.BFloat16(v)
		}
		lit, err = xlabuilder.NewArrayLiteral(pjFlat, dims...)
	default:
		lit, err = xlabuilder.NewArrayLiteralFromAny(flat, dims...)
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

// Parameter creates an input parameter for this function.
func (g *Graph) Parameter(name string, shape shapes.Shape, sharding *compute.ShardingSpec) (node compute.Value, err error) {
	index := len(g.in)
	if g.inputs != nil {
		op, err := g.tupleArgument(shape, name, index)
		if err != nil {
			return nil, err
		}
		g.in = append(g.in, op)
		return op, nil
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
func (g *Graph) UnaryFunc(x compute.Value, f func(*xlabuilder.Op) (*xlabuilder.Op, error)) (compute.Value, error) {
	result, err := f(g.xlaHandle(x))
	if err != nil {
		return nil, err
	}
	return g.newNode(result, x), nil
}

// BinaryFunc returns a node executing a binary function. f must be an xlabuilder function pointer.
func (g *Graph) BinaryFunc(x compute.Value, y compute.Value, f func(x *xlabuilder.Op, y *xlabuilder.Op) (*xlabuilder.Op, error)) (compute.Value, error) {
	result, err := f(g.xlaHandle(x), g.xlaHandle(y))
	if err != nil {
		return nil, err
	}
	return g.newNode(result, x, y), nil
}

// ReduceFunc returns a node executing a basic reduction. f must be an xlabuilder function pointer.
func (g *Graph) ReduceFunc(x compute.Value, axes []int, f func(*xlabuilder.Op, ...int) (*xlabuilder.Op, error)) (compute.Value, error) {
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
func (g *Graph) LogicalNot(x compute.Value) (compute.Value, error) {
	return g.UnaryFunc(x, xlabuilder.LogicalNot)
}

// Neg returns a node computing the negation of x.
func (g *Graph) Neg(x compute.Value) (compute.Value, error) {
	return g.UnaryFunc(x, xlabuilder.Neg)
}

// Add returns a node adding two nodes.
func (g *Graph) Add(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Add)
}

// Sub returns a node subtracting y from x.
func (g *Graph) Sub(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Sub)
}

// Mul returns a node multiplying two nodes.
func (g *Graph) Mul(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Mul)
}

// Div returns a node dividing x by y.
func (g *Graph) Div(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Div)
}

// Rem returns a node computing remainder of x divided by y.
func (g *Graph) Rem(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.Rem)
}

// Equal returns a boolean node checking if x == y.
func (g *Graph) Equal(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.Equal)
}

// NotEqual returns a boolean node checking if x != y.
func (g *Graph) NotEqual(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.NotEqual)
}

// LessThan returns a boolean node checking if x < y.
func (g *Graph) LessThan(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.LessThan)
}

// LessOrEqual returns a boolean node checking if x <= y.
func (g *Graph) LessOrEqual(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.LessOrEqual)
}

// GreaterThan returns a boolean node checking if x > y.
func (g *Graph) GreaterThan(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.GreaterThan)
}

// GreaterOrEqual returns a boolean node checking if x >= y.
func (g *Graph) GreaterOrEqual(x, y compute.Value) (compute.Value, error) {
	// TODO(paulchang): If both operands are floating-point, use TotalOrder comparisons.
	return g.BinaryFunc(x, y, xlabuilder.GreaterOrEqual)
}

// ShiftLeft returns a node shifting x left by y.
func (g *Graph) ShiftLeft(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.ShiftLeft)
}

// ShiftRightArithmetic returns a node shifting lhs right by rhs, preserving the sign bit.
func (g *Graph) ShiftRightArithmetic(lhs, rhs compute.Value) (compute.Value, error) {
	return g.BinaryFunc(lhs, rhs, xlabuilder.ShiftRightArithmetic)
}

// ShiftRightLogical returns a node shifting lhs right by rhs, ignoring the sign bit.
func (g *Graph) ShiftRightLogical(lhs, rhs compute.Value) (compute.Value, error) {
	return g.BinaryFunc(lhs, rhs, xlabuilder.ShiftRightLogical)
}

// BitwiseAnd returns a node computing bitwise AND of x and y.
func (g *Graph) BitwiseAnd(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseAnd)
}

// BitwiseOr returns a node computing bitwise OR of x and y.
func (g *Graph) BitwiseOr(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseOr)
}

// BitwiseXor returns a node computing bitwise XOR of x and y.
func (g *Graph) BitwiseXor(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.BitwiseXor)
}

// LogicalAnd returns a node computing logical AND of x and y.
func (g *Graph) LogicalAnd(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.LogicalAnd)
}

// LogicalOr returns a node computing logical OR of x and y.
func (g *Graph) LogicalOr(x, y compute.Value) (compute.Value, error) {
	return g.BinaryFunc(x, y, xlabuilder.LogicalOr)
}

// Reshape returns a reshape operator node.
func (g *Graph) Reshape(x compute.Value, dimensions ...int) (compute.Value, error) {
	xlaOp, err := xlabuilder.Reshape(g.xlaHandle(x), dimensions...)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// ConvertDType returns a cast/convert operator node.
func (g *Graph) ConvertDType(x compute.Value, dtype dtypes.DType) (compute.Value, error) {
	xlaDType := pjrtgx.ToPJDType(dtype)
	if xlaDType == pjtypes.InvalidDType {
		return nil, errors.Errorf("cannot convert %s to a XLA data type", dtype.String())
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
func (n *tuple) Element(i int) (compute.Value, error) {
	return n.element(i)
}

func (n *tuple) Size() int {
	// Note: this relies on gopjrt's shape tracking.
	return n.Node.op.Shape.TupleSize()
}

func (n *tuple) Unpack() ([]compute.Value, error) {
	nodes := make([]compute.Value, 0, n.Size())
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
func (g *Graph) Tuple(nodes []compute.Value) (backend.Tuple, error) {
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
func ToXLATuple(n compute.Value) backend.Tuple {
	if tpl, ok := n.(*tuple); ok {
		return tpl
	}
	return &tuple{Node: n.(*Node)}
}

// Slice returns a slice on a node.
func (g *Graph) Slice(x compute.Value, starts, limits, strides []int) (compute.Value, error) {
	sliceOp, err := xlabuilder.Slice(g.xlaHandle(x), starts, limits, strides)
	if err != nil {
		return nil, err
	}
	return g.newNode(sliceOp), nil
}

// BroadcastInDim broadcasts x to an output with the given shape.
func (g *Graph) BroadcastInDim(x compute.Value, shape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	xlaOp, err := xlabuilder.BroadcastInDim(g.xlaHandle(x), pjrtgx.ToShape(shape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Gather exposes the full XLA Gather operation.
func (g *Graph) Gather(x compute.Value, startIndices compute.Value, indexVectorAxis int, offsetAxes []int, collapsedSliceAxes []int, startIndexMap []int, sliceSizes []int, indicesAreSorted bool) (compute.Value, error) {
	xlaOp, err := xlabuilder.Gather(g.xlaHandle(x), g.xlaHandle(startIndices), indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// DynamicUpdateSlice updates a slice in an array.
func (g *Graph) DynamicUpdateSlice(operand, update compute.Value, startIndices []compute.Value) (compute.Value, error) {
	xlaStartIndices, err := g.xlaHandles(startIndices)
	if err != nil {
		return nil, err
	}
	xlaRes, err := xlabuilder.DynamicUpdateSlice(g.xlaHandle(operand), g.xlaHandle(update), xlaStartIndices)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaRes), nil
}

// DotGeneral returns a generic dot product node.
func (g *Graph) DotGeneral(lhs compute.Value, lhsContractingAxes, lhsBatchAxes []int, rhs compute.Value, rhsContractingAxes, rhsBatchAxes []int, config backend.DotGeneralConfig) (compute.Value, error) {
	xlaOp, err := xlabuilder.DotGeneral(
		g.xlaHandle(lhs), lhsContractingAxes, lhsBatchAxes,
		g.xlaHandle(rhs), rhsContractingAxes, rhsBatchAxes)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}

// Call returns a node that invokes a subgraph with the given result node.
func (g *Graph) Call(sg *backend.Subgraph, args ...compute.Value) (compute.Value, error) {
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
	var result compute.Value = g.newNode(xlaOp, subcomp)
	if _, ok := sg.Result.Node.(backend.Tuple); ok {
		// If the result node was a tuple, the subgraph's return value will also be a tuple.
		result = ToXLATuple(result)
	}
	return result, nil
}

// NewFunction creates a new named function within the builder.
func (g *Graph) NewFunction(name string) (backend.Function, error) {
	builder := g.builder.CreateSubBuilder(name)
	return newGraph(g.plat, g.bld, nil, nil, builder)
}

type subGraph struct {
	out   compute.Value
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
func (g *Graph) While(cond, body *backend.Subgraph, state compute.Value) (compute.Value, error) {
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
	var result compute.Value = g.newNode(xlaOp, condSG, bodySG)
	if _, ok := state.(backend.Tuple); ok {
		result = ToXLATuple(result)
	}
	return result, nil
}

// String representation of the graph.
func (g *Graph) String() string {
	return fmt.Sprintf("XLAGraph(%q):%p", g.builder.Name(), g.builder)
}
